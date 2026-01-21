"""
Evaluate models on nvidia/OpenMathInstruct-2 dataset (open-ended math problems).
"""
import os
import re
import json
import random
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.llm_utils import call_openrouter, get_openrouter_key_info

DATASET_NAME = "nvidia/OpenMathInstruct-2"
SEED = 42

PROBLEM_SOURCES = ["math", "gsm8k", "augmented_math", "augmented_gsm8k"]
SAMPLES_PER_SOURCE = 25

# (display_name, model_id, reasoning_config, max_tokens)
# reasoning_config can be: None, int (max_tokens), or str (effort: "high"/"medium"/"low")
MODELS = [
    ("llama", "meta-llama/llama-3.1-8b-instruct", None, 2048),
    ("grok", "x-ai/grok-4-fast", None, 4096),
    ("flash", "google/gemini-3-flash-preview", None, 4096),
    ("flash-think", "google/gemini-3-flash-preview", 5000, 15000),
    ("pro-think", "google/gemini-3-pro-preview", 5000, 15000),
    ("opus", "anthropic/claude-opus-4", "high", 15000),  # Oracle GT
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
    
    # Try to find \boxed{} first (common in math responses)
    boxed = extract_boxed(response_text)
    if boxed:
        return normalize_answer(boxed)
    
    # Try "Answer:" pattern
    match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1).strip())
    
    # Try "final answer is" pattern
    match = re.search(r'final answer is[:\s]*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1).strip())
    
    # Fallback to last line
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
    answer = re.sub(r'^\$+|\$+$', '', answer)  # Remove $ from LaTeX
    
    # Strip leading equation forms like y=, f(x)=, etc.
    answer = re.sub(r'^[a-z]\s*=\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^[a-z]\([a-z]\)\s*=\s*', '', answer, flags=re.IGNORECASE)
    
    # Strip degree symbols and common unit suffixes
    answer = re.sub(r'\^?\\?circ\b', '', answer)  # ^\circ or \circ
    answer = re.sub(r'°', '', answer)  # degree symbol
    answer = re.sub(r'\s*(inches?|feet|foot|mph|hours?|minutes?|seconds?|meters?|cm|km|miles?)\s*$', '', answer, flags=re.IGNORECASE)
    
    # Normalize \dfrac to \frac
    answer = answer.replace('\\dfrac', '\\frac')
    
    # Convert \frac{a}{b} to decimal (only for simple numeric fractions)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', eval_latex_fraction, answer)
    
    # Handle a/b fractions
    frac_match = re.match(r'^(-?\d+)/(-?\d+)$', answer.strip())
    if frac_match:
        try:
            answer = str(float(frac_match.group(1)) / float(frac_match.group(2)))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Only remove thousand-separator commas (pattern: digit,3digits)
    answer = re.sub(r'(\d),(\d{3})(?=,\d{3}|[^\d]|$)', r'\1\2', answer)
    answer = re.sub(r'\s+', '', answer)  # Remove whitespace
    answer = answer.lower()
    
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return f"{num:.10f}".rstrip('0').rstrip('.')
    except ValueError:
        return answer


def answers_match(predicted, expected):
    """Check if predicted answer matches expected."""
    if predicted is None:
        return False
    
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    if pred_norm == exp_norm:
        return True
    
    # Try numeric comparison with tolerance for floating point
    try:
        pred_num = float(pred_norm) if pred_norm else None
        exp_num = float(exp_norm) if exp_norm else None
        if pred_num is not None and exp_num is not None:
            # Use relative tolerance for larger numbers, absolute for small
            if abs(exp_num) > 1:
                return abs(pred_num - exp_num) / abs(exp_num) < 1e-4
            return abs(pred_num - exp_num) < 1e-6
    except ValueError:
        pass
    
    return False


def process_sample(sample, model_id, reasoning_config, max_tokens, api_key):
    """Process a single sample."""
    problem = sample['problem']
    expected = sample['expected_answer']
    idx = sample['idx']
    
    prompt = PROMPT_TEMPLATE.format(problem=problem)
    
    # Handle reasoning_config: None, int (max_tokens), or str (effort)
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
        run_id="openmath", record_id=str(idx), context="OpenMath"
    )
    elapsed_time = time.time() - start_time
    
    raw_response = response['content']
    predicted = extract_answer(raw_response)
    is_correct = answers_match(predicted, expected)
    
    return {
        'idx': idx,
        'problem': problem,
        'problem_source': sample.get('problem_source'),
        'expected': expected,
        'predicted': predicted,
        'is_correct': is_correct,
        'raw_response': raw_response,
        'token_usage': token_usage,
        'elapsed_time': elapsed_time
    }


def run_all_evals(samples, models, api_key):
    """Run evaluation for all models in parallel."""
    all_results = {name: [] for name, _, _, _ in models}
    total_tasks = len(samples) * len(models)
    completed = [0]  # Use list for mutable counter in closure
    
    print(f"\nRunning {total_tasks} tasks ({len(samples)} samples × {len(models)} models) in parallel...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {}
        for display_name, model_id, reasoning_config, max_tokens in models:
            for s in samples:
                future = executor.submit(process_sample, s, model_id, reasoning_config, max_tokens, api_key)
                futures[future] = (display_name, s)
        
        for future in as_completed(futures):
            display_name, sample = futures[future]
            completed[0] += 1
            try:
                result = future.result()
                result['model'] = display_name
                all_results[display_name].append(result)
                status = "+" if result['is_correct'] else "-"
                print(f"  [{completed[0]}/{total_tasks}] {display_name}: idx={result['idx']} {status}")
            except Exception as e:
                print(f"  [{completed[0]}/{total_tasks}] {display_name}: idx={sample['idx']} ERROR: {e}")
                all_results[display_name].append({
                    'idx': sample['idx'],
                    'problem': sample['problem'],
                    'problem_source': sample.get('problem_source'),
                    'expected': sample['expected_answer'],
                    'predicted': None,
                    'is_correct': False,
                    'error': str(e),
                    'model': display_name
                })
    
    return all_results


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    print(f"Loading samples from {DATASET_NAME}...")
    print(f"Sampling {SAMPLES_PER_SOURCE} per source: {PROBLEM_SOURCES}")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    rng = random.Random(SEED)
    
    samples_by_source = {source: [] for source in PROBLEM_SOURCES}
    for i, item in enumerate(dataset):
        source = item.get('problem_source')
        if source in samples_by_source:
            samples_by_source[source].append({
                'idx': i,
                'problem': item['problem'],
                'expected_answer': item['expected_answer'],
                'problem_source': source
            })
        if all(len(v) >= SAMPLES_PER_SOURCE * 10 for v in samples_by_source.values()):
            break
    
    samples = []
    for source in PROBLEM_SOURCES:
        pool = samples_by_source[source]
        selected = rng.sample(pool, min(SAMPLES_PER_SOURCE, len(pool)))
        samples.extend(selected)
    
    print(f"Selected {len(samples)} samples: {[s['problem_source'] for s in samples]}")
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    all_results_raw = run_all_evals(samples, MODELS, api_key)
    
    # Filter out samples where any model had an error/timeout (for analysis only)
    error_idxs = set()
    for model_name, results in all_results_raw.items():
        for r in results:
            if r.get('error') or r.get('predicted') is None:
                error_idxs.add(r['idx'])
    
    if error_idxs:
        print(f"\nExcluding {len(error_idxs)} samples with errors from analysis: {sorted(error_idxs)}")
    
    # Filtered results for analysis (excludes errors)
    all_results = {
        model_name: [r for r in results if r['idx'] not in error_idxs]
        for model_name, results in all_results_raw.items()
    }
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0) if key_info_end else 0
    total_cost = end_usage - start_usage
    
    print("\n" + "="*80)
    print("RESULTS: ACCURACY VS DATASET GT")
    print("="*80)
    
    for model_name, results in all_results.items():
        correct = sum(1 for r in results if r.get('is_correct'))
        total = len(results)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n{model_name}")
        print(f"  Overall: {correct}/{total} ({accuracy:.1f}%)")
        
        for source in PROBLEM_SOURCES:
            source_results = [r for r in results if r.get('problem_source') == source]
            if source_results:
                source_correct = sum(1 for r in source_results if r.get('is_correct'))
                source_total = len(source_results)
                source_acc = source_correct / source_total * 100
                print(f"    {source:20s} {source_correct}/{source_total} ({source_acc:.1f}%)")
    
    # Response time statistics
    print("\n" + "-"*80)
    print("RESPONSE TIME STATISTICS (seconds)")
    print("-"*80)
    
    for model_name, results in all_results.items():
        times = [r.get('elapsed_time', 0) for r in results if r.get('elapsed_time')]
        if not times:
            continue
        times = np.array(times)
        print(f"\n{model_name}")
        print(f"  Overall: p10={np.percentile(times, 10):.1f}  median={np.median(times):.1f}  mean={np.mean(times):.1f}  p90={np.percentile(times, 90):.1f}  max={np.max(times):.1f}")
        
        for source in PROBLEM_SOURCES:
            source_times = [r.get('elapsed_time', 0) for r in results if r.get('problem_source') == source and r.get('elapsed_time')]
            if source_times:
                t = np.array(source_times)
                print(f"    {source:20s} p10={np.percentile(t, 10):.1f}  median={np.median(t):.1f}  mean={np.mean(t):.1f}  p90={np.percentile(t, 90):.1f}  max={np.max(t):.1f}")
    
    # Build idx lookups for all models
    results_by_idx = {name: {r['idx']: r for r in results} for name, results in all_results.items()}
    all_idxs = list(next(iter(results_by_idx.values())).keys())
    
    def calc_acc(idxs, check_fn):
        if not idxs:
            return None, 0
        correct = sum(1 for idx in idxs if check_fn(idx))
        return correct, len(idxs)
    
    def print_metric(label, idxs, check_fn, indent=0):
        correct, total = calc_acc(idxs, check_fn)
        prefix = "  " * indent
        if total == 0:
            print(f"{prefix}{label}: N/A")
        else:
            print(f"{prefix}{label}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    def get_idxs_for_source(source):
        return [idx for idx in all_idxs if results_by_idx['llama'][idx]['problem_source'] == source]
    
    print("\n" + "-"*80)
    print("CROSS-MODEL ANALYSIS")
    print("-"*80)
    
    # Opus as oracle GT - compare each model against opus
    oracle = 'opus'
    other_models = [m for m in results_by_idx.keys() if m != oracle]
    
    if oracle in results_by_idx:
        print(f"\n[{oracle} as oracle GT]")
        for model_name in other_models:
            matches_oracle = lambda idx, m=model_name: answers_match(
                results_by_idx[m][idx]['predicted'], 
                results_by_idx[oracle][idx]['predicted']
            )
            print(f"\n  {model_name} accuracy ({oracle} as GT):")
            print_metric("Overall", all_idxs, matches_oracle, indent=2)
            for source in PROBLEM_SOURCES:
                print_metric(source, get_idxs_for_source(source), matches_oracle, indent=3)
    
    # Flash models as GT for llama
    print("\n[Flash models as GT for llama]")
    for judge_model in ['flash', 'flash-think']:
        if 'llama' in results_by_idx and judge_model in results_by_idx:
            matches_judge = lambda idx, j=judge_model: answers_match(
                results_by_idx['llama'][idx]['predicted'],
                results_by_idx[j][idx]['predicted']
            )
            print(f"\n  Llama accuracy ({judge_model} as GT):")
            print_metric("Overall", all_idxs, matches_judge, indent=2)
            for source in PROBLEM_SOURCES:
                print_metric(source, get_idxs_for_source(source), matches_judge, indent=3)
    
    # Llama stratified by judge correctness
    print("\n[Llama accuracy stratified by judge correctness vs dataset GT]")
    for judge_model in ['flash', 'flash-think', 'pro-think', 'opus']:
        if 'llama' in results_by_idx and judge_model in results_by_idx:
            judge_correct_idxs = [idx for idx in all_idxs if results_by_idx[judge_model][idx]['is_correct']]
            judge_wrong_idxs = [idx for idx in all_idxs if not results_by_idx[judge_model][idx]['is_correct']]
            llama_correct = lambda idx: results_by_idx['llama'][idx]['is_correct']
            
            print(f"\n  Llama when {judge_model} matches dataset GT:")
            print_metric("Overall", judge_correct_idxs, llama_correct, indent=2)
            for source in PROBLEM_SOURCES:
                source_idxs = [idx for idx in judge_correct_idxs if results_by_idx['llama'][idx]['problem_source'] == source]
                print_metric(source, source_idxs, llama_correct, indent=3)
            
            print(f"  Llama when {judge_model} does NOT match dataset GT:")
            if judge_wrong_idxs:
                print_metric("Overall", judge_wrong_idxs, llama_correct, indent=2)
                for source in PROBLEM_SOURCES:
                    source_idxs = [idx for idx in judge_wrong_idxs if results_by_idx['llama'][idx]['problem_source'] == source]
                    print_metric(source, source_idxs, llama_correct, indent=3)
            else:
                print(f"    N/A ({judge_model} got all correct)")
    
    # Per-sample breakdown
    print("\n" + "-"*120)
    print("Per-sample breakdown:")
    model_names = [name for name, _, _, _ in MODELS]
    header = f"{'idx':>6} | {'source':<16} | {'expected':<25}"
    for name in model_names:
        header += f" | {name:^17}"
    print(header)
    print("-" * len(header))
    
    first_model = model_names[0]
    for r in sorted(all_results[first_model], key=lambda x: (x['problem_source'], x['idx'])):
        idx = r['idx']
        expected = str(r['expected'])[:25]
        row = f"{idx:>6} | {r['problem_source']:<16} | {expected:<25}"
        for name in model_names:
            model_r = results_by_idx[name][idx]
            status = "+" if model_r['is_correct'] else "-"
            pred = str(model_r['predicted'])[:15]
            row += f" | {status} {pred:<15}"
        print(row)
    
    print(f"\nTotal cost: ${total_cost:.4f}")
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"openmath_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def build_summary(results):
        summary = {
            'correct': sum(1 for r in results if r.get('is_correct')),
            'total': len(results),
            'accuracy': sum(1 for r in results if r.get('is_correct')) / len(results) if results else 0,
            'by_source': {}
        }
        for source in PROBLEM_SOURCES:
            source_results = [r for r in results if r.get('problem_source') == source]
            if source_results:
                summary['by_source'][source] = {
                    'correct': sum(1 for r in source_results if r.get('is_correct')),
                    'total': len(source_results)
                }
        return summary
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'dataset': DATASET_NAME,
                'problem_sources': PROBLEM_SOURCES,
                'samples_per_source': SAMPLES_PER_SOURCE,
                'seed': SEED,
                'models': [(name, model_id, reasoning, max_tok) for name, model_id, reasoning, max_tok in MODELS],
                'temperature': TEMPERATURE
            },
            'results': all_results_raw,  # Save all results including errors
            'excluded_idxs': sorted(error_idxs),  # Samples excluded from analysis
            'summary': {model: build_summary(results) for model, results in all_results.items()},  # Summary excludes errors
            'total_cost': total_cost
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
