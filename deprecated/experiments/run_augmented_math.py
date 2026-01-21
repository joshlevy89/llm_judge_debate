"""
Evaluate models on augmented_math.json dataset.
"""
import os
import re
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_utils import call_openrouter, get_openrouter_key_info

MODELS = [
    ("llama", "meta-llama/llama-3.1-8b-instruct", None, 4096),
    ("pro-think", "google/gemini-3-pro-preview", 15000, 20000),
    ("opus", "anthropic/claude-opus-4", 15000, 20000),
]

TEMPERATURE = 0.0
MAX_THREADS = 50

PROMPT_TEMPLATE = """Solve this math problem. Show your work, then provide your final answer on the last line in the format: "Answer: <your answer>"

Important: Give ONLY the numerical value or mathematical expression as your answer. Do not include units (like "inches", "degrees", "mph"), do not include "y=" or "f(x)=" prefixes, and do not include degree symbols. If the answer is a simple fraction, output that rather than converting it to a decimal.

Problem: {problem}"""


def extract_boxed(text):
    """Extract content from \\boxed{...} handling nested braces."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1]
    return None


def extract_answer(response_text):
    """Extract the answer from model response."""
    if not response_text:
        return None
    
    boxed = extract_boxed(response_text)
    if boxed:
        return normalize_answer(boxed)
    
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1).strip())
    
    match = re.search(r'final answer is[:\s]*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1).strip())
    
    lines = response_text.strip().split('\n')
    if lines:
        return normalize_answer(lines[-1].strip())
    return None


def eval_latex_fraction(match):
    """Convert \frac{a}{b} to float, or a/b if not numeric."""
    num = match.group(1)
    denom = match.group(2)
    try:
        return str(float(num) / float(denom))
    except (ValueError, ZeroDivisionError):
        return f"{num}/{denom}"


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if not answer:
        return None
    
    answer = answer.strip()
    answer = re.sub(r'^\$+|\$+$', '', answer)
    answer = re.sub(r'^[a-z]\s*=\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^[a-z]\([a-z]\)\s*=\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\^?\\?circ\b', '', answer)
    answer = re.sub(r'°', '', answer)
    answer = re.sub(r'\s*(inches?|feet|foot|mph|hours?|minutes?|seconds?|meters?|cm|km|miles?)\s*$', '', answer, flags=re.IGNORECASE)
    answer = answer.replace('\\dfrac', '\\frac')
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', eval_latex_fraction, answer)
    
    frac_match = re.match(r'^(-?\d+)/(-?\d+)$', answer.strip())
    if frac_match:
        try:
            answer = str(float(frac_match.group(1)) / float(frac_match.group(2)))
        except (ValueError, ZeroDivisionError):
            pass
    
    answer = re.sub(r'(\d),(\d{3})(?=,\d{3}|[^\d]|$)', r'\1\2', answer)
    answer = re.sub(r'\s+', '', answer)
    answer = answer.lower()
    
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return f"{num:.10f}".rstrip('0').rstrip('.')
    except ValueError:
        return answer


def process_sample(sample, model_id, reasoning_config, max_tokens, api_key):
    """Process a single sample."""
    problem_id = sample['id']
    problem = sample['problem']
    
    prompt = PROMPT_TEMPLATE.format(problem=problem)
    
    reasoning_max_tokens = None
    reasoning_effort = None
    if isinstance(reasoning_config, int):
        reasoning_max_tokens = reasoning_config
    elif isinstance(reasoning_config, str):
        reasoning_effort = reasoning_config
    
    start_time = time.time()
    response, token_usage = call_openrouter(
        prompt, model_id, api_key, TEMPERATURE,
        max_tokens=max_tokens, reasoning_max_tokens=reasoning_max_tokens,
        reasoning_effort=reasoning_effort,
        run_id="augmented_math", record_id=problem_id, context="AugmentedMath"
    )
    elapsed_time = time.time() - start_time
    
    raw_response = response['content']
    predicted = extract_answer(raw_response)
    
    return {
        'id': problem_id,
        'problem': problem,
        'predicted': predicted,
        'raw_response': raw_response,
        'token_usage': token_usage,
        'elapsed_time': elapsed_time
    }


def run_all_evals(problems, models, api_key):
    """Run evaluation for all models in parallel."""
    all_results = {name: [] for name, _, _, _ in models}
    total_tasks = len(problems) * len(models)
    completed = [0]
    
    print(f"\nRunning {total_tasks} tasks ({len(problems)} problems × {len(models)} models) in parallel...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {}
        for display_name, model_id, reasoning_config, max_tokens in models:
            for p in problems:
                future = executor.submit(process_sample, p, model_id, reasoning_config, max_tokens, api_key)
                futures[future] = (display_name, p)
        
        for future in as_completed(futures):
            display_name, problem = futures[future]
            completed[0] += 1
            try:
                result = future.result()
                result['model'] = display_name
                all_results[display_name].append(result)
                print(f"  [{completed[0]}/{total_tasks}] {display_name}: {result['id']}")
            except Exception as e:
                print(f"  [{completed[0]}/{total_tasks}] {display_name}: {problem['id']} ERROR: {e}")
                all_results[display_name].append({
                    'id': problem['id'],
                    'problem': problem['problem'],
                    'predicted': None,
                    'error': str(e),
                    'model': display_name
                })
    
    return all_results


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    data_file = Path(__file__).parent / "augmented_math.json"
    print(f"Loading problems from {data_file}...")
    
    with open(data_file) as f:
        data = json.load(f)
    
    problems = []
    for level_name, level_problems in data['problems'].items():
        problems.extend(level_problems)
    
    print(f"Loaded {len(problems)} problems across {len(data['problems'])} levels")
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    all_results = run_all_evals(problems, MODELS, api_key)
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0) if key_info_end else 0
    total_cost = end_usage - start_usage
    
    print("\n" + "="*80)
    print("RESPONSE TIME STATISTICS (seconds)")
    print("="*80)
    
    for model_name, results in all_results.items():
        times = [r.get('elapsed_time', 0) for r in results if r.get('elapsed_time')]
        if not times:
            continue
        times = np.array(times)
        print(f"\n{model_name}")
        print(f"  p10={np.percentile(times, 10):.1f}  median={np.median(times):.1f}  mean={np.mean(times):.1f}  p90={np.percentile(times, 90):.1f}  max={np.max(times):.1f}")
    
    print(f"\nTotal cost: ${total_cost:.4f}")
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"augmented_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'dataset': 'augmented_math.json',
                'models': [(name, model_id, reasoning, max_tok) for name, model_id, reasoning, max_tok in MODELS],
                'temperature': TEMPERATURE
            },
            'results': all_results,
            'total_cost': total_cost
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
