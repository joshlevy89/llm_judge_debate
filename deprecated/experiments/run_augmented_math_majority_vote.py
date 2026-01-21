#!/usr/bin/env python3
"""
Run LLaMA 3.1 8B majority vote evaluation on augmented math questions.
Ground truth is established by Gemini 3 Flash.
"""
import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_utils import call_openrouter, get_openrouter_key_info

MODELS = {
    "llama": ("meta-llama/llama-3.1-8b-instruct", None, 2048),
    "flash": ("google/gemini-3-flash-preview", 5000, 20000),
    # "kimi": ("moonshotai/kimi-k2-thinking", 20000, 20000),
}

LLAMA_RUNS = 4
TEMPERATURE_LLAMA = 0.8
TOP_P_LLAMA = 0.95
TEMPERATURE_ORACLE = 0.0
MAX_THREADS = 500

PROMPT_TEMPLATE = """Solve this math problem. Show your work, then provide your final answer on the last line in the format: "Answer: <your answer>"

Details on formatting to check before outputting your final Answer:
- Simplify the answer to the simplest form.
- Do not use any formating (i.e. not latex formatting, no markdown, no $ etc)
- Do not include additional spaces within your answer. 
- Use a fraction rather than a decimal if the answer is a simple fraction.
- Do not include units.
- Do not include "y=" or "f(x)=" prefixes.
- Do not include any symbols (like degrees).
- Give ONLY the numerical value or mathematical expression as your answer. Do not include units (like "inches", "degrees", "mph"), do not include "y=" or "f(x)=" prefixes, and do not include degree symbols. If the answer is a simple fraction, output that rather than converting it to a decimal.

Problem: {problem}"""


def extract_boxed(text):
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


def eval_latex_fraction(match):
    num = match.group(1)
    denom = match.group(2)
    try:
        return str(float(num) / float(denom))
    except (ValueError, ZeroDivisionError):
        return f"{num}/{denom}"


def normalize_answer(answer):
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


def parse_answer(response_text):
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


def load_problems(input_file):
    with open(input_file) as f:
        data = json.load(f)
    
    problems = {}
    for q in data['questions']:
        idx = q['index']
        problems[idx] = {
            'idx': idx,
            'problem': q['augmented_problem'],
            'original_problem': q['original_problem'],
            'subject': q['subject'],
            'level': q['level']
        }
    
    return problems, data.get('metadata', {})


def call_model(problem, model_name, api_key, run_idx=None):
    model_id, reasoning_config, max_tokens = MODELS[model_name]
    
    prompt = PROMPT_TEMPLATE.format(problem=problem['problem'])
    
    reasoning_max_tokens = None
    reasoning_effort = None
    if isinstance(reasoning_config, int):
        reasoning_max_tokens = reasoning_config
    elif isinstance(reasoning_config, str):
        reasoning_effort = reasoning_config
    
    temperature = TEMPERATURE_LLAMA if model_name == "llama" else TEMPERATURE_ORACLE
    top_p = TOP_P_LLAMA if model_name == "llama" else None
    
    record_id = f"{problem['idx']}_{model_name}"
    if run_idx is not None:
        record_id += f"_{run_idx}"
    
    start_time = time.time()
    response, token_usage = call_openrouter(
        prompt, model_id, api_key, temperature,
        max_tokens=max_tokens, top_p=top_p, reasoning_max_tokens=reasoning_max_tokens,
        reasoning_effort=reasoning_effort,
        run_id="aug_math_majority", record_id=record_id, context="AugMathMajority"
    )
    elapsed_time = time.time() - start_time
    
    raw_response = response['content']
    parsed_answer = parse_answer(raw_response)
    
    return {
        'raw_response': raw_response,
        'parsed_answer': parsed_answer,
        'token_usage': token_usage,
        'elapsed_time': elapsed_time,
        'model_id': model_id,
        'temperature': temperature,
        'top_p': top_p
    }


def run_oracle(problems, api_key):
    oracle_results = {}
    total_tasks = len(problems)
    completed = [0]
    
    print(f"\nRunning Flash oracle ({total_tasks} tasks)...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(call_model, problem, 'flash', api_key): idx
                   for idx, problem in problems.items()}
        
        for future in as_completed(futures):
            idx = futures[future]
            completed[0] += 1
            try:
                result = future.result()
                oracle_results[idx] = result
                ans = str(result['parsed_answer'])[:50] if result['parsed_answer'] else "None"
                print(f"  [{completed[0]:3d}/{total_tasks}] flash idx={idx:3d} -> {ans}")
            except Exception as e:
                print(f"  [{completed[0]:3d}/{total_tasks}] flash idx={idx:3d} -> ERROR: {e}")
                oracle_results[idx] = {'error': str(e), 'parsed_answer': None}
    
    return oracle_results


def determine_ground_truth(oracle_results):
    ground_truth = {}
    valid_count = 0
    none_count = 0
    
    for idx, result in oracle_results.items():
        ans = result.get('parsed_answer')
        if ans is not None:
            ground_truth[idx] = ans
            valid_count += 1
        else:
            ground_truth[idx] = None
            none_count += 1
    
    return ground_truth, valid_count, none_count


def run_llama(problems, api_key):
    llama_results = {idx: [] for idx in problems}
    total_tasks = len(problems) * LLAMA_RUNS
    completed = [0]
    
    print(f"\nRunning LLaMA ({total_tasks} tasks: {len(problems)} problems x {LLAMA_RUNS} runs)...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {}
        for idx, problem in problems.items():
            for run_idx in range(LLAMA_RUNS):
                future = executor.submit(call_model, problem, 'llama', api_key, run_idx)
                futures[future] = (idx, run_idx)
        
        for future in as_completed(futures):
            idx, run_idx = futures[future]
            completed[0] += 1
            try:
                result = future.result()
                result['run_idx'] = run_idx
                llama_results[idx].append(result)
                ans = str(result['parsed_answer'])[:50] if result['parsed_answer'] else "None"
                print(f"  [{completed[0]:4d}/{total_tasks}] llama idx={idx:3d} run={run_idx} -> {ans}")
            except Exception as e:
                print(f"  [{completed[0]:4d}/{total_tasks}] llama idx={idx:3d} run={run_idx} -> ERROR: {e}")
                llama_results[idx].append({'error': str(e), 'parsed_answer': None, 'run_idx': run_idx})
    
    return llama_results


def main():
    parser = argparse.ArgumentParser(description="Run majority vote eval on augmented math")
    parser.add_argument("input_file", type=str, help="Path to augmented math JSON file")
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    input_path = Path(args.input_file)
    print(f"Loading problems from {input_path}...")
    problems, source_metadata = load_problems(input_path)
    print(f"Loaded {len(problems)} problems")
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    oracle_results = run_oracle(problems, api_key)
    
    ground_truth, valid_count, none_count = determine_ground_truth(oracle_results)
    
    print("\n" + "="*60)
    print("GROUND TRUTH ANALYSIS")
    print("="*60)
    print(f"Flash returned valid answer: {valid_count}")
    print(f"Flash returned None:         {none_count}")
    print(f"Total kept for analysis: {valid_count}/{len(problems)}")
    
    llama_results = run_llama(problems, api_key)
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0) if key_info_end else 0
    total_cost = end_usage - start_usage
    
    output_data = {}
    for idx, problem in problems.items():
        output_data[idx] = {
            'idx': idx,
            'ground_truth': ground_truth[idx],
            'problem': problem['problem'],
            'original_problem': problem['original_problem'],
            'subject': problem['subject'],
            'level': problem['level'],
            'oracle_result': oracle_results.get(idx),
            'llama_results': sorted(llama_results[idx], key=lambda x: x.get('run_idx', 0))
        }
    
    output_file = input_path.parent / f"majority_vote_{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'source_file': str(input_path),
                'source_metadata': source_metadata,
                'models': {k: list(v) for k, v in MODELS.items()},
                'llama_runs': LLAMA_RUNS,
                'temperature_llama': TEMPERATURE_LLAMA,
                'top_p_llama': TOP_P_LLAMA,
                'temperature_oracle': TEMPERATURE_ORACLE,
                'prompt_template': PROMPT_TEMPLATE
            },
            'ground_truth_summary': {
                'valid_count': valid_count,
                'none_count': none_count,
                'total_problems': len(problems)
            },
            'results': output_data,
            'total_cost': total_cost
        }, f, indent=2)
    
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
