"""
Generate augmented MATH questions and validate them.

Step 1 of the augmented math pipeline:
1. Generate questions from MATH dataset seeds using Grok 4.1 Fast (no thinking)
2. Validate each question using DeepSeek v3.2 (no thinking)
3. Assign ground truth where generator and validator agree (both non-null)

Output: augmented_math_YYYYMMDD_HHMMSS.json
"""

import os
import json
import random
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
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

GENERATOR_MODEL = "x-ai/grok-4.1-fast"
VALIDATOR_MODEL = "deepseek/deepseek-v3.2"

LEVELS = [1, 2, 3, 4, 5]
SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

# TODO: For Step 2, set specific level/subject and increase count
QUESTIONS_PER_CATEGORY = 25
NUM_MC_OPTIONS = 10
MAX_THREADS = 1000

GENERATE_PROMPT = """Create a new math problem inspired by this seed problem, solve it, and generate multiple choice options.

SEED PROBLEM:
{seed_question}

INSTRUCTIONS:
1. Create a NEW problem inspired by the seed (don't just change numbers - use a different approach)
2. Solve your new problem step by step
3. Generate {num_distractors} plausible wrong answers (common mistakes students make)"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_question",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The new math problem statement"},
                "solution": {"type": "string", "description": "Step by step solution"},
                "correct_answer": {"type": "string", "description": "The final answer"},
                "wrong_answers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Plausible but incorrect answers"
                }
            },
            "required": ["question", "solution", "correct_answer", "wrong_answers"],
            "additionalProperties": False
        }
    }
}


def load_math_dataset():
    print("Loading MATH dataset...")
    organized = {level: {subj: [] for subj in SUBJECTS} for level in LEVELS}
    
    for subject in SUBJECTS:
        print(f"  Loading {subject}...")
        dataset = load_dataset("EleutherAI/hendrycks_math", subject)
        
        for split in ["train", "test"]:
            for item in dataset[split]:
                level_str = item.get("level", "")
                try:
                    level = int(level_str.replace("Level ", "")) if "Level " in level_str else None
                except ValueError:
                    continue
                if level in LEVELS:
                    organized[level][subject].append(item["problem"])
    
    return organized


def generate_question(seed, api_key):
    prompt = GENERATE_PROMPT.format(seed_question=seed, num_distractors=NUM_MC_OPTIONS - 1)
    
    response, usage = call_openrouter(
        prompt, GENERATOR_MODEL, api_key,
        temperature=0.7,
        max_tokens=3000,
        reasoning_enabled=False,
        response_format=RESPONSE_SCHEMA
    )
    
    if response.get("finish_reason") == "length":
        return None, "truncated", usage
    
    content = response.get("content") or ""
    if not content:
        return None, "empty", usage
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, f"json_error: {e.msg}", usage
    
    question = data.get("question")
    correct = data.get("correct_answer")
    wrong = data.get("wrong_answers", [])
    
    if not question or not correct or len(wrong) < NUM_MC_OPTIONS - 1:
        return None, "missing_fields", usage
    
    return {
        "question": question,
        "solution": data.get("solution", ""),
        "correct_answer": correct,
        "wrong_answers": wrong[:NUM_MC_OPTIONS - 1]
    }, "ok", usage


def build_validator_prompt(question, options_text):
    response_format = SHARED_PROMPTS["response_format_prompt"].replace("{N}", str(len(options_text.split("\n"))))
    return QA_PROMPTS["qa_prompt_template"].format(
        question=question,
        options_text=options_text,
        response_format_prompt=response_format
    )


def validate_question(question_text, options, api_key):
    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = build_validator_prompt(question_text, options_text)
    
    response, usage = call_openrouter(
        prompt, VALIDATOR_MODEL, api_key,
        temperature=0.5,
        max_tokens=15000,
        reasoning_enabled=False
    )
    
    content = response.get("content") or ""
    parsed = parse_answer(content, lenient=True)
    
    return parsed["answer"], usage


def process_task(level, subject, seed, api_key, rng_seed):
    rng = random.Random(rng_seed)
    gen_usage = {}
    val_usage = {}
    
    try:
        result, status, gen_usage = generate_question(seed, api_key)
        if not result:
            return None, status, (level, subject), gen_usage, val_usage
        
        options = result["wrong_answers"] + [result["correct_answer"]]
        rng.shuffle(options)
        generator_idx = options.index(result["correct_answer"])
        
        validator_idx, val_usage = validate_question(result["question"], options, api_key)
        
        # GT is assigned only if generator and validator agree and both non-null
        if generator_idx is not None and validator_idx is not None and generator_idx == validator_idx:
            ground_truth = generator_idx
        else:
            ground_truth = None
        
        return {
            "level": level,
            "subject": subject,
            "question": result["question"],
            "solution": result["solution"],
            "options": options,
            "generator_idx": generator_idx,
            "validator_idx": validator_idx,
            "ground_truth": ground_truth,
            "seed_question": seed
        }, "ok", (level, subject), gen_usage, val_usage
        
    except Exception as e:
        return None, f"exception: {type(e).__name__}: {str(e)[:50]}", (level, subject), gen_usage, val_usage


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate and validate augmented math questions")
    parser.add_argument("--max-threads", type=int, default=MAX_THREADS, help="Max parallel threads")
    parser.add_argument("--questions-per-category", type=int, default=QUESTIONS_PER_CATEGORY)
    parser.add_argument("--level", type=int, help="Specific level to generate (for Step 2)")
    parser.add_argument("--subject", type=str, help="Specific subject to generate (for Step 2)")
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    levels = [args.level] if args.level else LEVELS
    subjects = [args.subject] if args.subject else SUBJECTS
    
    math_data = load_math_dataset()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
    tasks = []
    base_rng = random.Random(42)
    for level in levels:
        for subject in subjects:
            seeds = math_data[level][subject]
            selected = base_rng.sample(seeds, min(args.questions_per_category, len(seeds)))
            for seed in selected:
                task_seed = base_rng.randint(0, 2**31)
                tasks.append((level, subject, seed, task_seed))
    
    total = len(tasks)
    print(f"\nGenerating {total} questions ({len(levels)} levels x {len(subjects)} subjects x {args.questions_per_category} each)")
    print(f"Generator: {GENERATOR_MODEL}, Validator: {VALIDATOR_MODEL}")
    print(f"Threads: {args.max_threads}")
    
    results = []
    completed = 0
    failed = 0
    failure_counts = {}
    total_gen_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    total_val_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {
            executor.submit(process_task, level, subject, seed, api_key, task_seed): (level, subject)
            for level, subject, seed, task_seed in tasks
        }
        
        for future in as_completed(futures):
            try:
                result, status, (level, subject), gen_usage, val_usage = future.result()
                tag = f"L{level}/{subject}"
                
                for k in total_gen_usage:
                    total_gen_usage[k] += gen_usage.get(k, 0)
                    total_val_usage[k] += val_usage.get(k, 0)
                
                if result:
                    results.append(result)
                    completed += 1
                    gt_status = "GT" if result["ground_truth"] is not None else "no-GT"
                    print(f"  [OK {completed}/{total}] {tag} gen:{result['generator_idx']} val:{result['validator_idx']} ({gt_status})")
                else:
                    failed += 1
                    failure_counts[status] = failure_counts.get(status, 0) + 1
                    print(f"  [FAIL {failed}/{total}] {tag}: {status}")
            except Exception as e:
                failed += 1
                failure_counts["exception"] = failure_counts.get("exception", 0) + 1
                print(f"  [FAIL {failed}/{total}] {type(e).__name__}")
    
    results.sort(key=lambda x: (x["level"], x["subject"], x["question"]))
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    total_cost = end_usage - start_usage
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), "data", f"augmented_math_{timestamp}.json")
    
    output_data = {
        "metadata": {
            "generator_model": GENERATOR_MODEL,
            "validator_model": VALIDATOR_MODEL,
            "levels": levels,
            "subjects": subjects,
            "questions_per_category": args.questions_per_category,
            "num_mc_options": NUM_MC_OPTIONS,
            "timestamp": timestamp,
            "total_generated": len(results),
            "total_failed": failed,
            "failure_counts": failure_counts,
            "generator_usage": total_gen_usage,
            "validator_usage": total_val_usage,
            "total_cost": total_cost
        },
        "questions": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    gt_count = sum(1 for r in results if r["ground_truth"] is not None)
    
    print(f"\n{'='*50}")
    print(f"Generated {len(results)}/{total} questions ({failed} failed)")
    print(f"Ground truth assigned: {gt_count}/{len(results)} ({100*gt_count/len(results):.1f}%)")
    if failure_counts:
        print(f"Failure breakdown: {failure_counts}")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
