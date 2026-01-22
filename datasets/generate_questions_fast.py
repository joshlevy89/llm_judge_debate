"""
Fast question generation script - simplified and cost-optimized.

Generates augmented MATH questions using a single API call per question
with reasoning disabled for ~30-100x cost reduction vs original script.
"""

import os
import json
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import call_openrouter, get_openrouter_key_info

MODEL = "x-ai/grok-4.1-fast"
MAX_TOKENS = 3000
TEMPERATURE = 0.7

LEVELS = [1, 2, 3, 4, 5]
SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]
QUESTIONS_PER_CATEGORY = 10
NUM_MC_OPTIONS = 10
MAX_THREADS = 400

COMBINED_PROMPT = """Create a new math problem inspired by this seed problem, solve it, and generate multiple choice options.

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
    """Single API call to generate question, solve it, and create distractors."""
    prompt = COMBINED_PROMPT.format(
        seed_question=seed,
        num_distractors=NUM_MC_OPTIONS - 1
    )
    
    response, usage = call_openrouter(
        prompt, MODEL, api_key,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_enabled=False,
        response_format=RESPONSE_SCHEMA
    )
    
    # Check for truncation
    if response.get("finish_reason") == "length":
        return None, "truncated"
    
    content = response.get("content") or ""
    if not content:
        return None, "empty"
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        # Log first 200 chars to understand what's failing
        preview = content[:200].replace('\n', '\\n') if content else "empty"
        return None, f"json_error: {e.msg} at pos {e.pos}, preview: {preview}"
    
    question = data.get("question")
    correct = data.get("correct_answer")
    wrong = data.get("wrong_answers", [])
    
    if not question or not correct or len(wrong) < NUM_MC_OPTIONS - 1:
        return None, "missing_fields"
    
    return {
        "question": question,
        "solution": data.get("solution", ""),
        "correct_answer": correct,
        "wrong_answers": wrong[:NUM_MC_OPTIONS - 1]
    }, "ok"


def process_task(level, subject, seed, api_key, rng_seed):
    """Process a single task - no retries. Returns (result, status)."""
    rng = random.Random(rng_seed)
    
    try:
        result, status = generate_question(seed, api_key)
        if not result:
            return None, status, (level, subject)
        
        # Shuffle options
        options = result["wrong_answers"] + [result["correct_answer"]]
        rng.shuffle(options)
        correct_idx = options.index(result["correct_answer"])
        
        return {
            "level": level,
            "subject": subject,
            "question": result["question"],
            "solution": result["solution"],
            "options": options,
            "correct_idx": correct_idx,
            "correct_answer": result["correct_answer"],
            "seed_question": seed
        }, "ok", (level, subject)
    except Exception as e:
        return None, f"exception: {type(e).__name__}", (level, subject)


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    math_data = load_math_dataset()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
    # Build tasks
    tasks = []
    base_rng = random.Random(42)
    for level in LEVELS:
        for subject in SUBJECTS:
            seeds = math_data[level][subject]
            selected = base_rng.sample(seeds, min(QUESTIONS_PER_CATEGORY, len(seeds)))
            for seed in selected:
                task_seed = base_rng.randint(0, 2**31)
                tasks.append((level, subject, seed, task_seed))
    
    total = len(tasks)
    print(f"\nGenerating {total} questions with {MAX_THREADS} threads...")
    
    results = []
    completed = 0
    failed = 0
    failure_counts = {}
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_task, level, subject, seed, api_key, task_seed): (level, subject)
            for level, subject, seed, task_seed in tasks
        }
        
        for future in as_completed(futures):
            try:
                result, status, (level, subject) = future.result()
                tag = f"L{level}/{subject}"
                if result:
                    results.append(result)
                    completed += 1
                    print(f"  [OK {completed}/{total}] {tag}")
                else:
                    failed += 1
                    failure_counts[status] = failure_counts.get(status, 0) + 1
                    print(f"  [FAIL {failed}/{total}] {tag}: {status}")
            except Exception as e:
                failed += 1
                failure_counts["exception"] = failure_counts.get("exception", 0) + 1
                print(f"  [FAIL {failed}/{total}] {type(e).__name__}")
    
    results.sort(key=lambda x: (x["level"], x["subject"], x["question"]))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), "data", f"questions_fast_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    cost = end_usage - start_usage
    
    total_failed = sum(failure_counts.values())
    print(f"\n{'='*50}")
    print(f"Generated {len(results)}/{total} questions ({total_failed} failed)")
    if failure_counts:
        print(f"Failure breakdown: {failure_counts}")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
