import re
import json
import time
import requests
from pathlib import Path
from threading import Thread
from requests.adapters import HTTPAdapter
from config.config_general import REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF_FACTOR, HTTP_POOL_CONNECTIONS, HTTP_POOL_MAXSIZE

_session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=HTTP_POOL_CONNECTIONS,
    pool_maxsize=HTTP_POOL_MAXSIZE,
    max_retries=0
)
_session.mount('https://', adapter)
_session.mount('http://', adapter)

class RequestWithTimeout:
    def __init__(self):
        self.result = None
        self.exception = None
    
    def make_request(self, url, headers, json_data):
        try:
            self.result = _session.post(url, headers=headers, json=json_data, timeout=30)
        except Exception as e:
            self.exception = e

def _make_openrouter_request(prompt, model_name, api_key, temperature=0.0, max_tokens=None, top_p=None, reasoning_effort=None, reasoning_max_tokens=None, reasoning_enabled=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": 42,
        "order": ["fireworks"]
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    if top_p is not None:
        data["top_p"] = top_p
    
    if reasoning_effort or reasoning_max_tokens:
        reasoning_config = {}
        if reasoning_effort:
            reasoning_config["effort"] = reasoning_effort
        if reasoning_max_tokens:
            reasoning_config["max_tokens"] = reasoning_max_tokens
        data["reasoning"] = reasoning_config
    
    if reasoning_enabled is not None:
        if reasoning_enabled:
            data["reasoning"] = {'enabled': True}
        else:
            data["reasoning"] = {'enabled': False}

    # print(data)
    
    req = RequestWithTimeout()
    thread = Thread(target=req.make_request, args=(url, headers, data))
    thread.daemon = True
    thread.start()
    thread.join(timeout=REQUEST_TIMEOUT)
    
    if thread.is_alive():
        raise requests.Timeout(f"Request exceeded {REQUEST_TIMEOUT} second timeout")
    
    if req.exception:
        raise req.exception
    
    # print(req.result.json())

    if req.result is None:
        raise Exception("Request returned no result")

    
    if req.result.status_code != 200:
        try:
            error_json = req.result.json()
            error_msg = error_json.get('error', {}).get('message', str(error_json))
        except:
            error_msg = req.result.text[:500]
        raise Exception(f"API returned status {req.result.status_code}: {error_msg}")
    
    try:
        return req.result.json()
    except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as e:
        raise Exception(
            f"JSON decode error: {str(e)}\n"
            f"Status code: {req.result.status_code}\n"
            f"Response preview: {req.result.text[:500]}"
        )

def call_openrouter(prompt, model_name, api_key, temperature=0.0, reasoning_effort=None, reasoning_max_tokens=None, reasoning_enabled=None, max_tokens=None, top_p=None, run_id=None, record_id=None, context=None):
    for attempt in range(MAX_RETRIES + 1):
        try:
            response_json = _make_openrouter_request(prompt, model_name, api_key, temperature, max_tokens, top_p, reasoning_effort, reasoning_max_tokens, reasoning_enabled)
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                message = response_json['choices'][0]['message']
                return {
                    'content': message.get('content'),
                    'reasoning': message.get('reasoning'),
                    'reasoning_details': message.get('reasoning_details')
                }, response_json.get('usage', {})
            
            error_text = response_json.get('error', {}).get('message', 'Unknown error') if 'error' in response_json else 'No response from model'
            raise Exception(f"API Error: {error_text}")
            
        except requests.Timeout:
            if attempt < MAX_RETRIES:
                print(f"Timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}) - Run: {run_id} - {context or 'Request'}")
                time.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
            else:
                raise Exception(f"Request timeout after {MAX_RETRIES + 1} attempts")
        except Exception as e:
            if "API Error:" in str(e) or "Request timeout" in str(e):
                raise
            raise Exception(str(e))

# def log_progress(status_type, count, total, run_id, record_id, api_key, start_usage, error=None, is_correct=None):
#     key_info = get_openrouter_key_info(api_key)
#     current_usage = key_info.get('data', {}).get('usage', 0) if key_info else 0
#     cost = current_usage - start_usage if current_usage else 0
    
#     if status_type == "completed":
#         extra = f" - Correct: {is_correct}" if is_correct is not None else ""
#         print(f"Completed {count}/{total} - Run ID: {run_id} - Record ID: {record_id}{extra} - Cost: ${cost:.6f}")
#     else:
#         error_msg = str(error)[:100] if error else "Unknown error"
#         print(f"Failed {count}/{total} - Run ID: {run_id} - Record ID: {record_id} - Error: {error_msg} - Cost: ${cost:.6f}")
    
#     return cost


def log_progress(status_message, count, total, run_id, record_id, api_key, start_usage, error=None, is_correct=None):
    key_info = get_openrouter_key_info(api_key)
    current_usage = key_info.get('data', {}).get('usage', 0) if key_info else 0
    cost = current_usage - start_usage if current_usage else 0
    
    extra = f" - Correct: {is_correct}"
    error_msg = f" - Error: {str(error)[:100] if error else None}"
    print(f"{status_message} {count}/{total} - Run ID: {run_id} - Record ID: {record_id} - Cost: ${cost:.6f}{extra}{error_msg}")
    
def get_openrouter_key_info(api_key):
    if not api_key:
        return None
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Warning] Could not fetch OpenRouter key info: {e}")
    return None

def parse_answer(response_text, lenient=True):
    parsed = {
        'is_valid': False,
        'answer': None,
        'confidence': None,
        'reasoning': None
    }
    
    if lenient:
        search_text = response_text
    else:
        final_answer_match = re.search(r'<BEGIN FINAL ANSWER>(.*?)</?END FINAL ANSWER>', response_text, re.DOTALL | re.IGNORECASE)
        if not final_answer_match:
            return parsed
        search_text = final_answer_match.group(1)
    
    answer_match = re.search(r'Answer:\s*(\d+)', search_text, re.IGNORECASE)
    if answer_match:
        parsed['answer'] = int(answer_match.group(1))
        parsed['is_valid'] = True
    
    confidence_match = re.search(r'Confidence:\s*(\d+)(?:\.\d+)?%?', search_text, re.IGNORECASE)
    if confidence_match:
        parsed['confidence'] = int(confidence_match.group(1))
    
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\s*$|\Z)', search_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        parsed['reasoning'] = reasoning_match.group(1).strip()
    
    return parsed

