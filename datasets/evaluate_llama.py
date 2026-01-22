"""
Evaluate augmented math questions with Llama 3.1 8B judge.

Step 2 of the augmented math pipeline:
1. Load augmented_math_*.json (output from generate_questions.py)
2. Filter to questions with ground truth (generator-validator agreement)
3. Either binarize (2-choice) or use all 10 options
4. Run Llama 3.1 8B on each question
5. Record accuracy

Output: llama_{mode}_{timestamp}.json
"""

import os
import json
import random
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import call_openrouter, get_openrouter_key_info, parse_answer

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
with open(os.path.join(PROMPTS_DIR, "qa_prompts.yaml")) as f:
    QA_PROMPTS = yaml.safe_load(f)
with open(os.path.join(PROMPTS_DIR, "shared_prompts.yaml")) as f:
    SHARED_PROMPTS = yaml.safe_load(f)

JUDGE_MODEL = "meta-llama/llama-3.1-8b-instruct"
MAX_THREADS = 200


def build_prompt(question, options_text, num_options):
    response_format = SHARED_PROMPTS["response_format_prompt"].replace("{N}", str(num_options))
    return QA_PROMPTS["qa_prompt_template"].format(
        question=question,
        options_text=options_text,
        response_format_prompt=response_format
    )


def binarize_question(question_data, rng):
    """Convert 10-option question to 2-option (correct + 1 random incorrect)."""
    gt_idx = question_data["ground_truth"]
    correct_answer = question_data["options"][gt_idx]
    
    wrong_options = [opt for i, opt in enumerate(question_data["options"]) if i != gt_idx]
    wrong_answer = rng.choice(wrong_options)
    
    binary_options = [correct_answer, wrong_answer]
    rng.shuffle(binary_options)
    binary_gt_idx = binary_options.index(correct_answer)
    
    return binary_options, binary_gt_idx


def evaluate_question(question_text, options, api_key):
    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = build_prompt(question_text, options_text, len(options))
    
    response, usage = call_openrouter(
        prompt, JUDGE_MODEL, api_key,
        temperature=0.5,
        max_tokens=20000
    )
    
    content = response.get("content") or ""
    parsed = parse_answer(content, lenient=True)
    
    # Validate answer is in valid range [0, N-1]
    answer = parsed["answer"]
    if answer is not None and (answer < 0 or answer >= len(options)):
        answer = None
    
    return answer, parsed["confidence"], content, usage


def process_question_binary(idx, question_data, api_key, rng_seed):
    """Process question in binary (2-choice) mode."""
    rng = random.Random(rng_seed)
    
    try:
        binary_options, binary_gt_idx = binarize_question(question_data, rng)
        llama_answer, confidence, raw_content, usage = evaluate_question(
            question_data["question"], binary_options, api_key
        )
        
        is_correct = llama_answer == binary_gt_idx if llama_answer is not None else None
        
        return {
            "idx": idx,
            "level": question_data["level"],
            "subject": question_data["subject"],
            "question": question_data["question"],
            "options": binary_options,
            "gt_idx": binary_gt_idx,
            "llama_answer": llama_answer,
            "llama_confidence": confidence,
            "is_correct": is_correct,
            "raw_content": raw_content,
            "usage": usage
        }, "ok"
        
    except Exception as e:
        return {
            "idx": idx,
            "level": question_data["level"],
            "subject": question_data["subject"],
            "question": question_data["question"],
            "error": str(e)
        }, f"exception: {type(e).__name__}"


def process_question_ten(idx, question_data, api_key, rng_seed):
    """Process question in 10-choice mode."""
    try:
        options = question_data["options"]
        gt_idx = question_data["ground_truth"]
        
        llama_answer, confidence, raw_content, usage = evaluate_question(
            question_data["question"], options, api_key
        )
        
        is_correct = llama_answer == gt_idx if llama_answer is not None else None
        
        return {
            "idx": idx,
            "level": question_data["level"],
            "subject": question_data["subject"],
            "question": question_data["question"],
            "options": options,
            "gt_idx": gt_idx,
            "llama_answer": llama_answer,
            "llama_confidence": confidence,
            "is_correct": is_correct,
            "raw_content": raw_content,
            "usage": usage
        }, "ok"
        
    except Exception as e:
        return {
            "idx": idx,
            "level": question_data["level"],
            "subject": question_data["subject"],
            "question": question_data["question"],
            "error": str(e)
        }, f"exception: {type(e).__name__}"


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluate augmented math with Llama judge")
    parser.add_argument("input_file", help="Path to augmented_math_*.json")
    parser.add_argument("--mode", choices=["binary", "ten"], default="binary",
                        help="Evaluation mode: 'binary' (2-choice) or 'ten' (10-choice)")
    parser.add_argument("--max-threads", type=int, default=MAX_THREADS, help="Max parallel threads")
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    with open(args.input_file) as f:
        data = json.load(f)
    
    all_questions = data["questions"]
    input_metadata = data["metadata"]
    
    questions_with_gt = [q for q in all_questions if q["ground_truth"] is not None]
    print(f"Loaded {len(all_questions)} questions, {len(questions_with_gt)} have ground truth")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Mode: {args.mode} ({'2-choice' if args.mode == 'binary' else '10-choice'})")
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
    results = []
    completed = 0
    failed = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    base_rng = random.Random(42)
    task_seeds = [base_rng.randint(0, 2**31) for _ in questions_with_gt]
    
    process_fn = process_question_binary if args.mode == "binary" else process_question_ten
    
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {
            executor.submit(process_fn, idx, q, api_key, task_seeds[idx]): idx
            for idx, q in enumerate(questions_with_gt)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result, status = future.result()
                results.append(result)
                
                if "usage" in result:
                    for k in total_usage:
                        total_usage[k] += result["usage"].get(k, 0)
                
                completed += 1
                if status == "ok":
                    tag = f"L{result['level']}/{result['subject']}"
                    correct_str = "correct" if result.get("is_correct") else ("null" if result.get("llama_answer") is None else "wrong")
                    print(f"  [OK {completed}/{len(questions_with_gt)}] {tag} llama:{result.get('llama_answer')} gt:{result.get('gt_idx')} ({correct_str})")
                else:
                    failed += 1
                    print(f"  [FAIL {completed}/{len(questions_with_gt)}] {status}")
                    
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {type(e).__name__}: {str(e)[:50]}")
    
    results.sort(key=lambda x: x["idx"])
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    total_cost = end_usage - start_usage
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), "data", f"llama_{args.mode}_{timestamp}.json")
    
    error_count = sum(1 for r in results if "error" in r)
    null_count = sum(1 for r in results if r.get("llama_answer") is None and "error" not in r)
    valid_results = [r for r in results if "is_correct" in r and r["is_correct"] is not None]
    correct_count = sum(1 for r in valid_results if r["is_correct"])
    
    num_choices = 2 if args.mode == "binary" else 10
    output_data = {
        "metadata": {
            "input_file": os.path.basename(args.input_file),
            "input_metadata": input_metadata,
            "judge_model": JUDGE_MODEL,
            "mode": args.mode,
            "num_choices": num_choices,
            "timestamp": timestamp,
            "total_evaluated": len(results),
            "total_errors": error_count,
            "total_nulls": null_count,
            "total_valid": len(valid_results),
            "total_correct": correct_count,
            "accuracy": correct_count / len(valid_results) if valid_results else None,
            "judge_usage": total_usage,
            "total_cost": total_cost
        },
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results breakdown ({args.mode} mode):")
    print(f"  Total with GT:    {len(questions_with_gt)}")
    print(f"  API errors:       {error_count} ({100*error_count/len(results):.1f}%)")
    print(f"  Parse nulls:      {null_count} ({100*null_count/len(results):.1f}%)")
    print(f"  Valid answers:    {len(valid_results)} ({100*len(valid_results)/len(results):.1f}%)")
    print(f"Llama accuracy: {correct_count}/{len(valid_results)} ({100*correct_count/len(valid_results):.1f}%)" if valid_results else "No valid results")
    print(f"Random baseline: {100/num_choices:.1f}%")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
