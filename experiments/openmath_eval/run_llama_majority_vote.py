"""
Run LLaMA 3.1 8B majority vote evaluation with Opus 4.5 and Gemini Pro as oracles.
Ground truth is established when Opus and Gemini agree.
"""
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm_utils import call_openrouter, get_openrouter_key_info

RESULTS_FILE = Path(__file__).parent / "results" / "openmath_eval_20260118_123416.json"

MODELS = {
    "llama": ("meta-llama/llama-3.1-8b-instruct", None, 2048),
    "opus": ("anthropic/claude-opus-4", "high", 15000),
    "gemini": ("google/gemini-3-pro-preview", 5000, 15000),
}

LLAMA_RUNS = 16
TEMPERATURE_LLAMA = 0.8
TOP_P_LLAMA = 0.95
TEMPERATURE_ORACLE = 0.0
MAX_THREADS = 50

# TODO: Fill in the prompt template
PROMPT_TEMPLATE = """Solve this math problem. Show your work, then provide your final answer on the last line in the format: "Answer: <your answer>"

Details on formatting to check before outputting your final Answer:
- Do not use any formating (i.e. not latex formatting, no markdown, no $ etc)
- Do not include additional spaces within your answer. 
- Use a fraction rather than a decimal if the answer is a simple fraction.
- Do not include units.
- Do not include "y=" or "f(x)=" prefixes.
- Do not include any symbols (like degrees).
- Give ONLY the numerical value or mathematical expression as your answer. Do not include units (like "inches", "degrees", "mph"), do not include "y=" or "f(x)=" prefixes, and do not include degree symbols. If the answer is a simple fraction, output that rather than converting it to a decimal.

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


def eval_latex_fraction(match):
    """Convert \\frac{a}{b} to float, or a/b if not numeric."""
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


def parse_answer(response_text):
    """Extract and normalize the answer from model response."""
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


def answers_match(a, b):
    """Check if two answers match."""
    if a is None or b is None:
        return False
    
    norm_a = normalize_answer(a)
    norm_b = normalize_answer(b)
    
    if norm_a == norm_b:
        return True
    
    try:
        num_a = float(norm_a) if norm_a else None
        num_b = float(norm_b) if norm_b else None
        if num_a is not None and num_b is not None:
            if abs(num_b) > 1:
                return abs(num_a - num_b) / abs(num_b) < 1e-4
            return abs(num_a - num_b) < 1e-6
    except ValueError:
        pass
    
    return False


def load_problems():
    """Load problem data from the results file."""
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    
    problems = {}
    for result in data['results']['llama']:
        idx = result['idx']
        if idx not in problems:
            problems[idx] = {
                'idx': idx,
                'problem': result['problem'],
                'problem_source': result['problem_source'],
                'expected': result['expected']
            }
    
    return problems


def call_model(problem, model_name, api_key, run_idx=None):
    """Call a model and return the response with parsed answer."""
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
        run_id="llama_majority", record_id=record_id, context="LLamaMajority"
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


def run_oracle_models(problems, api_key):
    """Run opus and gemini once each on all problems."""
    oracle_results = {'opus': {}, 'gemini': {}}
    total_tasks = len(problems) * 2
    completed = [0]
    
    print(f"\nRunning oracle models ({total_tasks} tasks)...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {}
        for idx, problem in problems.items():
            for model_name in ['opus', 'gemini']:
                future = executor.submit(call_model, problem, model_name, api_key)
                futures[future] = (idx, model_name)
        
        for future in as_completed(futures):
            idx, model_name = futures[future]
            completed[0] += 1
            try:
                result = future.result()
                oracle_results[model_name][idx] = result
                ans = str(result['parsed_answer'])[:50] if result['parsed_answer'] else "None"
                print(f"  [{completed[0]:3d}/{total_tasks}] {model_name:6s} idx={idx:5d} -> {ans}")
            except Exception as e:
                print(f"  [{completed[0]:3d}/{total_tasks}] {model_name:6s} idx={idx:5d} -> ERROR: {e}")
                oracle_results[model_name][idx] = {'error': str(e), 'parsed_answer': None}
    
    return oracle_results


def determine_ground_truth(oracle_results):
    """Find idxs where opus and gemini agree. Returns dict of idx -> GT answer."""
    ground_truth = {}
    agree_count = 0
    disagree_count = 0
    none_count = 0
    
    all_idxs = set(oracle_results['opus'].keys()) | set(oracle_results['gemini'].keys())
    
    for idx in all_idxs:
        opus_ans = oracle_results['opus'].get(idx, {}).get('parsed_answer')
        gemini_ans = oracle_results['gemini'].get(idx, {}).get('parsed_answer')
        
        if opus_ans is None or gemini_ans is None:
            ground_truth[idx] = None
            none_count += 1
        elif answers_match(opus_ans, gemini_ans):
            ground_truth[idx] = opus_ans
            agree_count += 1
        else:
            ground_truth[idx] = None
            disagree_count += 1
    
    return ground_truth, agree_count, disagree_count, none_count


def run_llama(problems, api_key):
    """Run llama LLAMA_RUNS times on all problems."""
    llama_results = {idx: [] for idx in problems}
    total_tasks = len(problems) * LLAMA_RUNS
    completed = [0]
    
    print(f"\nRunning LLaMA ({total_tasks} tasks: {len(problems)} problems × {LLAMA_RUNS} runs)...")
    
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
                print(f"  [{completed[0]:4d}/{total_tasks}] llama idx={idx:5d} run={run_idx:2d} -> {ans}")
            except Exception as e:
                print(f"  [{completed[0]:4d}/{total_tasks}] llama idx={idx:5d} run={run_idx:2d} -> ERROR: {e}")
                llama_results[idx].append({'error': str(e), 'parsed_answer': None, 'run_idx': run_idx})
    
    return llama_results


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    print(f"Loading problems from {RESULTS_FILE}...")
    problems = load_problems()
    print(f"Loaded {len(problems)} unique problems")
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    oracle_results = run_oracle_models(problems, api_key)
    
    ground_truth, agree_count, disagree_count, none_count = determine_ground_truth(oracle_results)
    
    print("\n" + "="*60)
    print("GROUND TRUTH ANALYSIS")
    print("="*60)
    print(f"Opus and Gemini agree (kept):    {agree_count}")
    print(f"Opus and Gemini disagree (dropped): {disagree_count}")
    print(f"One or both returned None (dropped): {none_count}")
    print(f"Total kept for analysis: {agree_count}/{len(problems)}")
    
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
            'problem_source': problem['problem_source'],
            'expected_original': problem['expected'],
            'oracle_results': {
                'opus': oracle_results['opus'].get(idx),
                'gemini': oracle_results['gemini'].get(idx)
            },
            'llama_results': sorted(llama_results[idx], key=lambda x: x.get('run_idx', 0))
        }
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"llama_majority_vote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'source_file': str(RESULTS_FILE),
                'models': MODELS,
                'llama_runs': LLAMA_RUNS,
                'temperature_llama': TEMPERATURE_LLAMA,
                'top_p_llama': TOP_P_LLAMA,
                'temperature_oracle': TEMPERATURE_ORACLE,
                'prompt_template': PROMPT_TEMPLATE
            },
            'ground_truth_summary': {
                'agree_count': agree_count,
                'disagree_count': disagree_count,
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
