import json
import yaml
import random
import string
import time
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.llm_utils import call_openrouter, get_openrouter_key_info, parse_answer, log_progress
from utils.dataset_utils import select_questions_and_options, format_options, load_dataset_unified
from utils.shared_utils import generate_run_id, load_prompts


def load_specific_question_idxs(specific_question_idxs):
    if specific_question_idxs is None:
        return None
    if isinstance(specific_question_idxs, str) and specific_question_idxs.endswith('.txt'):
        file_path = Path(specific_question_idxs)
        if not file_path.is_absolute():
            file_path = Path(__file__).parent.parent / file_path
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    return specific_question_idxs

def format_qa_prompt(question, options, num_choices, prompt_format=None):
    prompt_template = load_prompts('qa')
    response_format_prompt = load_prompts('shared', variant=prompt_format)
    options_text = format_options(options)
    number_choices = ', '.join(str(i) for i in range(num_choices))
    
    return prompt_template.format(
        question=question,
        options_text=options_text,
        letter_choices=number_choices,
        response_format_prompt=response_format_prompt
    )

def check_qa_exists(question_idx, model_name, prompt, qa_results_path):
    if not qa_results_path.exists():
        return False
    
    with open(qa_results_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if (record.get('success') is not False and
                record.get('question_idx') == question_idx and
                record.get('config', {}).get('model_name') == model_name and
                record.get('prompt') == prompt):
                return True
    return False

def normalize_whitespace(text):
    return ' '.join(text.split())

def get_existing_qa_keys(qa_results_path):
    existing_qa = set()
    if not qa_results_path.exists():
        return existing_qa
    
    with open(qa_results_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record.get('success') is not False:
                config = record.get('config', {})
                key = (
                    record.get('question_idx'),
                    config.get('model_name'),
                    normalize_whitespace(record.get('prompt', '')),
                    config.get('reasoning_effort'),
                    config.get('reasoning_max_tokens')
                )
                existing_qa.add(key)
    return existing_qa

def filter_existing_questions(question_idxs, questions_data, model_name, num_choices, existing_qa, reasoning_effort=None, reasoning_max_tokens=None, prompt_format=None):
    missing_idxs = []
    
    for idx, q_data in zip(question_idxs, questions_data):
        prompt = format_qa_prompt(q_data['question'], q_data['options'], num_choices, prompt_format)
        key = (idx, model_name, normalize_whitespace(prompt), reasoning_effort, reasoning_max_tokens)
        
        if key not in existing_qa:
            missing_idxs.append(idx)
    
    return missing_idxs

def process_qa_question(q_data, prompt_template_str, api_key, config, run_id, run_datetime, model_name, temperature, max_tokens, reasoning_effort, reasoning_max_tokens, reasoning_enabled, num_choices, prompt_format=None):
    from utils.shared_utils import generate_run_id
    
    record_id = generate_run_id()
    prompt = format_qa_prompt(q_data['question'], q_data['options'], num_choices, prompt_format)
    
    response, token_usage = call_openrouter(
        prompt,
        model_name,
        api_key,
        temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_max_tokens=reasoning_max_tokens,
        reasoning_enabled=reasoning_enabled,
        run_id=run_id,
        record_id=record_id,
        context="QA"
    )
    
    raw_model_response = response['content']
    lenient_parsing = config.get('lenient_parsing', True)
    parsed_model_response = parse_answer(raw_model_response, lenient=lenient_parsing)
    
    return {
        'run_id': run_id,
        'record_id': record_id,
        'datetime': run_datetime,
        'config': config,
        'prompt_template': prompt_template_str,
        'question_idx': q_data['original_idx'],
        'question': q_data['question'],
        'options': q_data['options'],
        'correct_idx': q_data['correct_idx'],
        'prompt': prompt,
        'success': True,
        'error_message': None,
        'raw_model_response': raw_model_response,
        'internal_model_reasoning': response.get('reasoning'),
        'internal_model_reasoning_details': response.get('reasoning_details'),
        'parsed_model_response': parsed_model_response,
        'token_usage': token_usage
    }

def run_qa_for_questions(question_idxs, model_name, temperature, max_tokens, reasoning_effort, reasoning_max_tokens, dataset_config, num_choices, api_key, max_threads, run_id=None, qa_results_path=None, random_seed=None, reasoning_enabled=None, prompt_format=None):
    if run_id is None:
        run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))
    
    if qa_results_path is None:
        qa_results_path = Path('results') / 'qa' / 'qa_results.jsonl'
    
    qa_results_path.parent.mkdir(parents=True, exist_ok=True)
    run_datetime = datetime.now().isoformat()
    
    dataset, _ = load_dataset_unified(
        dataset_config['dataset_name'], 
        dataset_config.get('dataset_subset'),
        dataset_config.get('dataset_split'),
        dataset_config.get('dataset_path')
    )
    questions_data = select_questions_and_options(
        dataset_config['dataset_name'], 
        dataset, 
        len(question_idxs), 
        num_choices, 
        None,
        question_idxs
    )
    
    prompt_template = load_prompts('qa')
    config = {**dataset_config, 'model_name': model_name, 'temperature': temperature, 'max_tokens': max_tokens, 'reasoning_effort': reasoning_effort, 'reasoning_max_tokens': reasoning_max_tokens, 'reasoning_enabled': reasoning_enabled, 'num_choices': num_choices, 'random_seed': random_seed, 'prompt_format': prompt_format}
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_qa_question, q_data, prompt_template, api_key, config, run_id, run_datetime, model_name, temperature, max_tokens, reasoning_effort, reasoning_max_tokens, reasoning_enabled, num_choices, prompt_format): q_data
            for q_data in questions_data
        }
        
        with open(qa_results_path, 'a') as f:
            for future in as_completed(futures):
                q_data = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    is_correct = result['parsed_model_response']['answer'] == result['correct_idx'] if result['parsed_model_response']['answer'] is not None else None
                    log_progress("completed", completed, len(questions_data), result['run_id'], result['record_id'], api_key, start_usage, is_correct=is_correct)
                except Exception as e:
                    failed += 1
                    error_trace = traceback.format_exc()
                    error_record_id = generate_run_id()
                    result = {
                        'success': False,
                        'error_message': error_trace,
                        'run_id': run_id,
                        'record_id': error_record_id,
                        'datetime': run_datetime,
                        'config': config,
                        'question_idx': q_data['original_idx'],
                        'question': q_data['question'],
                        'options': q_data['options'],
                        'correct_idx': q_data['correct_idx']
                    }
                    log_progress("failed", failed, len(questions_data), run_id, error_record_id, api_key, start_usage, error=error_trace)
                
                f.write(json.dumps(result) + '\n')
                f.flush()
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0)
    
    return {
        'run_id': run_id,
        'completed': completed,
        'failed': failed,
        'cost': end_usage - start_usage
    }

