"""
Generate augmented MATH questions and validate them.

Step 1 of the augmented math pipeline:
1. Generate questions from MATH dataset seeds using Grok 4.1 Fast (no thinking)
2. Validate each question using DeepSeek v3.2 (no thinking)
3. Assign ground truth where generator and validator agree (both non-null)
4. Detect and mark near-duplicate questions

Three seed modes:
- Evolving seeds (default): Each generated question replaces its seed in the pool,
  causing the question space to "drift" over generations for natural diversity.
- Batch per seed (--batch-seeds): Generate all questions for each seed in one call,
  explicitly instructing the model to create diverse, non-duplicate questions.
- Static seeds (--static-seeds): All questions generated from original MATH seeds,
  one at a time, relying on temperature for diversity.

Output includes provenance tracking:
- original_seeds: list of original MATH questions with seed_id
- Each generated question has original_seed_id and parent_id for lineage tracking

Each question has validity flags:
- generation_ok: Question was successfully generated
- validation_ok: Validator returned a non-null answer
- gt_assigned: Generator and validator agreed on the answer
- is_duplicate: Question is a near-duplicate of an earlier question
- valid: Composite flag (generation_ok and gt_assigned and not is_duplicate)

Output: augmented_math_YYYYMMDD_HHMMSS.json
"""

import os
import json
import random
import argparse
import threading
import time
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv

import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import call_openrouter, get_openrouter_key_info, parse_answer
from check_duplicates import find_similar_pairs, get_duplicate_clusters

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
with open(os.path.join(PROMPTS_DIR, "qa_prompts.yaml")) as f:
    QA_PROMPTS = yaml.safe_load(f)
with open(os.path.join(PROMPTS_DIR, "shared_prompts.yaml")) as f:
    SHARED_PROMPTS = yaml.safe_load(f)

GENERATOR_MODEL = "x-ai/grok-4.1-fast"
VALIDATOR_MODEL = "deepseek/deepseek-v3.2"

ALL_LEVELS = [1, 2, 3, 4, 5]
ALL_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

NUM_MC_OPTIONS = 10
MAX_THREADS = 1000
DEFAULT_DUPLICATE_THRESHOLD = 0.99

GENERATE_PROMPT_SINGLE = """Create a new math problem inspired by this seed problem, solve it, and generate multiple choice options.

SEED PROBLEM:
{seed_question}

INSTRUCTIONS:
1. Create a NEW problem inspired by the seed (don't just change numbers - use a different approach)
2. Solve your new problem step by step
3. Generate {num_distractors} plausible wrong answers (common mistakes students make)"""

GENERATE_PROMPT_BATCH = """Create {num_questions} different math problems inspired by this seed problem. For each, solve it and generate multiple choice options.

SEED PROBLEM:
{seed_question}

INSTRUCTIONS:
1. Create {num_questions} NEW problems, each inspired by the seed but DISTINCT from each other
2. Vary the approach, context, and difficulty - don't just change numbers
3. For each problem, solve it step by step
4. For each problem, generate {num_distractors} plausible wrong answers (common mistakes students make)
5. IMPORTANT: Ensure all {num_questions} problems are meaningfully different from each other"""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The math problem statement"},
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

RESPONSE_SCHEMA_SINGLE = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_question",
        "strict": True,
        "schema": QUESTION_SCHEMA
    }
}

RESPONSE_SCHEMA_BATCH = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_questions_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": QUESTION_SCHEMA,
                    "description": "List of generated math questions"
                }
            },
            "required": ["questions"],
            "additionalProperties": False
        }
    }
}


def load_math_dataset():
    print("Loading MATH dataset...")
    organized = {level: {subj: [] for subj in ALL_SUBJECTS} for level in ALL_LEVELS}
    
    for subject in ALL_SUBJECTS:
        print(f"  Loading {subject}...")
        dataset = load_dataset("EleutherAI/hendrycks_math", subject)
        
        for split in ["train", "test"]:
            for item in dataset[split]:
                level_str = item.get("level", "")
                try:
                    level = int(level_str.replace("Level ", "")) if "Level " in level_str else None
                except ValueError:
                    continue
                if level in ALL_LEVELS:
                    organized[level][subject].append(item["problem"])
    
    return organized


def generate_question_single(seed, api_key):
    prompt = GENERATE_PROMPT_SINGLE.format(seed_question=seed, num_distractors=NUM_MC_OPTIONS - 1)
    
    response, usage = call_openrouter(
        prompt, GENERATOR_MODEL, api_key,
        temperature=1.0,
        max_tokens=3000,
        reasoning_enabled=False,
        response_format=RESPONSE_SCHEMA_SINGLE
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


def generate_questions_batch(seed, num_questions, api_key):
    prompt = GENERATE_PROMPT_BATCH.format(
        seed_question=seed, 
        num_questions=num_questions,
        num_distractors=NUM_MC_OPTIONS - 1
    )
    
    response, usage = call_openrouter(
        prompt, GENERATOR_MODEL, api_key,
        temperature=1.0,
        max_tokens=3000 * num_questions,
        reasoning_enabled=False,
        response_format=RESPONSE_SCHEMA_BATCH
    )
    
    if response.get("finish_reason") == "length":
        return [], "truncated", usage
    
    content = response.get("content") or ""
    if not content:
        return [], "empty", usage
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return [], f"json_error: {e.msg}", usage
    
    questions_data = data.get("questions", [])
    valid_questions = []
    
    for q in questions_data:
        question = q.get("question")
        correct = q.get("correct_answer")
        wrong = q.get("wrong_answers", [])
        
        if question and correct and len(wrong) >= NUM_MC_OPTIONS - 1:
            valid_questions.append({
                "question": question,
                "solution": q.get("solution", ""),
                "correct_answer": correct,
                "wrong_answers": wrong[:NUM_MC_OPTIONS - 1]
            })
    
    status = "ok" if valid_questions else "no_valid_questions"
    return valid_questions, status, usage


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


def validate_and_build_result(generated, level, subject, original_seed_id, parent_id, rng, api_key):
    """Validate a generated question and build the result dict."""
    options = generated["wrong_answers"] + [generated["correct_answer"]]
    rng.shuffle(options)
    generator_idx = options.index(generated["correct_answer"])
    
    validator_idx, val_usage = validate_question(generated["question"], options, api_key)
    
    validation_ok = validator_idx is not None
    gt_assigned = generator_idx is not None and validator_idx is not None and generator_idx == validator_idx
    ground_truth = generator_idx if gt_assigned else None
    
    return {
        "level": level,
        "subject": subject,
        "question": generated["question"],
        "solution": generated["solution"],
        "options": options,
        "generator_idx": generator_idx,
        "validator_idx": validator_idx,
        "ground_truth": ground_truth,
        "original_seed_id": original_seed_id,
        "parent_id": parent_id,
        "generation_ok": True,
        "validation_ok": validation_ok,
        "gt_assigned": gt_assigned,
        "is_duplicate": False,
        "duplicate_of": None,
        "failure_reason": None,
        "valid": gt_assigned  # Will be updated after duplicate detection
    }, val_usage


def build_failed_result(level, subject, original_seed_id, parent_id, failure_reason):
    """Build a result dict for a failed generation."""
    return {
        "level": level,
        "subject": subject,
        "question": None,
        "solution": None,
        "options": None,
        "generator_idx": None,
        "validator_idx": None,
        "ground_truth": None,
        "original_seed_id": original_seed_id,
        "parent_id": parent_id,
        "generation_ok": False,
        "validation_ok": False,
        "gt_assigned": False,
        "is_duplicate": False,
        "duplicate_of": None,
        "failure_reason": failure_reason,
        "valid": False
    }


def process_task_single(level, subject, seed, original_seed_id, parent_id, api_key, rng_seed):
    """Generate and validate a single question."""
    rng = random.Random(rng_seed)
    gen_usage = {}
    val_usage = {}
    
    try:
        result, status, gen_usage = generate_question_single(seed, api_key)
        if not result:
            failed = build_failed_result(level, subject, original_seed_id, parent_id, status)
            return failed, status, (level, subject), gen_usage, val_usage, None
        
        validated, val_usage = validate_and_build_result(
            result, level, subject, original_seed_id, parent_id, rng, api_key
        )
        return validated, "ok", (level, subject), gen_usage, val_usage, result["question"]
        
    except Exception as e:
        failure_reason = f"exception: {type(e).__name__}: {str(e)[:50]}"
        failed = build_failed_result(level, subject, original_seed_id, parent_id, failure_reason)
        return failed, failure_reason, (level, subject), gen_usage, val_usage, None


def generate_batch_task(level, subject, seed, original_seed_id, num_questions, api_key):
    """Generate multiple questions from one seed (no validation)."""
    try:
        generated_list, status, gen_usage = generate_questions_batch(seed, num_questions, api_key)
        return generated_list, status, (level, subject, original_seed_id), gen_usage
    except Exception as e:
        return [], f"exception: {type(e).__name__}: {str(e)[:50]}", (level, subject, original_seed_id), {}


def validate_single_task(generated, level, subject, original_seed_id, api_key, rng_seed):
    """Validate a single generated question."""
    rng = random.Random(rng_seed)
    try:
        validated, val_usage = validate_and_build_result(
            generated, level, subject, original_seed_id, None, rng, api_key
        )
        return validated, "ok", (level, subject), val_usage
    except Exception as e:
        failure_reason = f"val_exception: {type(e).__name__}: {str(e)[:50]}"
        failed = build_failed_result(level, subject, original_seed_id, None, failure_reason)
        failed["question"] = generated.get("question")
        failed["solution"] = generated.get("solution")
        failed["generation_ok"] = True  # Generation succeeded, validation failed
        return failed, failure_reason, (level, subject), {}


def process_task_evolving(seed_pool, pool_lock, original_seed_ids, idx, api_key, rng_seed, parent_questions):
    """Process task for evolving seeds mode."""
    with pool_lock:
        level, subject, seed = seed_pool[idx]
        original_seed_id = original_seed_ids[idx]
        parent_id = parent_questions.get(idx)
    
    result, status, cat, gen_usage, val_usage, new_question = process_task_single(
        level, subject, seed, original_seed_id, parent_id, api_key, rng_seed
    )
    return result, status, cat, gen_usage, val_usage, new_question, idx


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate and validate augmented math questions")
    parser.add_argument("--num-questions", type=int, required=True, help="Total questions to generate")
    parser.add_argument("--max-threads", type=int, default=MAX_THREADS, help="Max parallel threads")
    parser.add_argument("--level", type=int, nargs='+', help="Specific level(s) to filter (optional, e.g., --level 1 2 3)")
    parser.add_argument("--subject", type=str, nargs='+', help="Specific subject(s) to filter (optional)")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--batch-seeds", action="store_true", help="Use batch seeds mode (generate all questions per seed in one call)")
    mode_group.add_argument("--static-seeds", action="store_true", help="Use static seeds mode (all questions from original MATH seeds, one at a time)")
    parser.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUPLICATE_THRESHOLD,
                        help=f"Similarity threshold for duplicate detection (default: {DEFAULT_DUPLICATE_THRESHOLD})")
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    levels = args.level if args.level else ALL_LEVELS
    subjects = args.subject if args.subject else ALL_SUBJECTS
    
    math_data = load_math_dataset()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
    base_rng = random.Random(42)
    
    # Build original seeds list with seed_id
    original_seeds = []
    seed_id_map = {}
    for level in levels:
        for subject in subjects:
            for seed_text in math_data[level][subject]:
                seed_id = len(original_seeds)
                original_seeds.append({
                    "seed_id": seed_id,
                    "level": level,
                    "subject": subject,
                    "question": seed_text
                })
                seed_id_map[(level, subject, seed_text)] = seed_id
    
    total_seeds = len(original_seeds)
    if total_seeds == 0:
        raise ValueError("No seeds found for specified levels/subjects")
    
    total = args.num_questions
    inflation = total / total_seeds if total_seeds > 0 else 0
    questions_per_seed = math.ceil(inflation) if inflation > 0 else 1
    
    if args.static_seeds:
        seed_mode = "static"
    elif args.batch_seeds:
        seed_mode = "batch"
    else:
        seed_mode = "evolving"
    
    print(f"\nGenerating {total} questions from {total_seeds} seeds ({inflation:.1f}x inflation)")
    print(f"Seed mode: {seed_mode}")
    if seed_mode == "batch":
        print(f"Questions per seed: {questions_per_seed}")
    print(f"Levels: {levels}, Subjects: {subjects}")
    print(f"Generator: {GENERATOR_MODEL} (temp=1.0), Validator: {VALIDATOR_MODEL}")
    print(f"Duplicate threshold: {args.duplicate_threshold}")
    print(f"Threads: {args.max_threads}")
    
    results = []
    completed = 0
    failed = 0
    failure_counts = {}
    total_gen_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    total_val_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    if seed_mode == "batch":
        # Batch mode: generate all questions per seed, validate all in parallel
        base_rng.shuffle(original_seeds)
        
        with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
            # Submit all generation tasks upfront
            gen_futures = set()
            val_futures = set()
            future_info = {}
            gen_completed = 0
            
            for seed_info in original_seeds:
                future = executor.submit(
                    generate_batch_task,
                    seed_info["level"], seed_info["subject"], seed_info["question"],
                    seed_info["seed_id"], questions_per_seed, api_key
                )
                gen_futures.add(future)
                future_info[future] = ("gen", seed_info)
            
            # Process both gen and val futures as they complete
            while gen_futures or val_futures:
                all_futures = gen_futures | val_futures
                done = {f for f in all_futures if f.done()}
                if not done:
                    time.sleep(0.01)
                    continue
                
                for future in done:
                    task_type, info = future_info.pop(future)
                    
                    if task_type == "gen":
                        gen_futures.remove(future)
                        seed_info = info
                        gen_completed += 1
                        try:
                            generated_list, status, (level, subject, original_seed_id), gen_usage = future.result()
                            tag = f"L{level}/{subject}"
                            
                            for k in total_gen_usage:
                                total_gen_usage[k] += gen_usage.get(k, 0)
                            
                            if generated_list:
                                print(f"  [GEN {gen_completed}/{total_seeds}] {tag} seed_id={original_seed_id}: {len(generated_list)} questions")
                                for generated in generated_list:
                                    task_seed = base_rng.randint(0, 2**31)
                                    val_future = executor.submit(
                                        validate_single_task,
                                        generated, level, subject, original_seed_id, api_key, task_seed
                                    )
                                    val_futures.add(val_future)
                                    future_info[val_future] = ("val", (level, subject))
                            else:
                                for _ in range(questions_per_seed):
                                    failed_result = build_failed_result(level, subject, original_seed_id, None, f"gen:{status}")
                                    results.append(failed_result)
                                failed += questions_per_seed
                                failure_counts[f"gen:{status}"] = failure_counts.get(f"gen:{status}", 0) + questions_per_seed
                                print(f"  [GEN FAIL] {tag} seed_id={seed_info['seed_id']}: {status}")
                        except Exception as e:
                            for _ in range(questions_per_seed):
                                failed_result = build_failed_result(level, subject, seed_info["seed_id"], None, "gen:exception")
                                results.append(failed_result)
                            failed += questions_per_seed
                            failure_counts["gen:exception"] = failure_counts.get("gen:exception", 0) + questions_per_seed
                            print(f"  [GEN FAIL] seed_id={seed_info['seed_id']}: {type(e).__name__}")
                    
                    else:  # val
                        val_futures.remove(future)
                        level, subject = info
                        try:
                            result, status, _, val_usage = future.result()
                            tag = f"L{level}/{subject}"
                            
                            for k in total_val_usage:
                                total_val_usage[k] += val_usage.get(k, 0)
                            
                            results.append(result)
                            if result.get("valid") or result.get("gt_assigned"):
                                completed += 1
                                gt_status = "GT" if result["ground_truth"] is not None else "no-GT"
                                print(f"  [VAL {completed}] {tag} gen:{result['generator_idx']} val:{result['validator_idx']} ({gt_status})")
                            else:
                                failed += 1
                                failure_counts[f"val:{status}"] = failure_counts.get(f"val:{status}", 0) + 1
                                print(f"  [VAL FAIL] {tag}: {status}")
                        except Exception as e:
                            failed_result = build_failed_result(level, subject, None, None, "val:exception")
                            results.append(failed_result)
                            failed += 1
                            failure_counts["val:exception"] = failure_counts.get("val:exception", 0) + 1
                            print(f"  [VAL FAIL] {type(e).__name__}")
    
    elif seed_mode == "static":
        # Static seeds mode: generate one question at a time from original seeds
        seed_pool = [(s["level"], s["subject"], s["question"], s["seed_id"]) for s in original_seeds]
        reps_per_seed = (total + len(seed_pool) - 1) // len(seed_pool)
        selected_pool = (seed_pool * reps_per_seed)[:total]
        base_rng.shuffle(selected_pool)
        
        tasks = []
        for level, subject, seed, seed_id in selected_pool:
            task_seed = base_rng.randint(0, 2**31)
            tasks.append((level, subject, seed, seed_id, task_seed))
        
        with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
            futures = {
                executor.submit(process_task_single, level, subject, seed, seed_id, None, api_key, task_seed): (level, subject)
                for level, subject, seed, seed_id, task_seed in tasks
            }
            
            for future in as_completed(futures):
                level, subject = futures[future]
                try:
                    result, status, (level, subject), gen_usage, val_usage, _ = future.result()
                    tag = f"L{level}/{subject}"
                    
                    for k in total_gen_usage:
                        total_gen_usage[k] += gen_usage.get(k, 0)
                        total_val_usage[k] += val_usage.get(k, 0)
                    
                    results.append(result)
                    if result.get("valid") or result.get("gt_assigned"):
                        completed += 1
                        gt_status = "GT" if result["ground_truth"] is not None else "no-GT"
                        print(f"  [OK {completed}/{total}] {tag} gen:{result['generator_idx']} val:{result['validator_idx']} ({gt_status})")
                    else:
                        failed += 1
                        failure_counts[status] = failure_counts.get(status, 0) + 1
                        print(f"  [FAIL {failed}/{total}] {tag}: {status}")
                except Exception as e:
                    failed_result = build_failed_result(level, subject, None, None, "exception")
                    results.append(failed_result)
                    failed += 1
                    failure_counts["exception"] = failure_counts.get("exception", 0) + 1
                    print(f"  [FAIL {failed}/{total}] {type(e).__name__}")
    
    else:
        # Evolving seeds mode: seed pool updates as questions are generated
        seed_pool = [(s["level"], s["subject"], s["question"]) for s in original_seeds]
        original_seed_ids = [s["seed_id"] for s in original_seeds]
        parent_questions = {}
        pool_lock = threading.Lock()
        pool_size = len(seed_pool)
        next_question_id = [0]
        
        with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
            pending_futures = set()
            submitted = 0
            
            initial_batch = min(args.max_threads, total)
            for _ in range(initial_batch):
                idx = base_rng.randint(0, pool_size - 1)
                task_seed = base_rng.randint(0, 2**31)
                future = executor.submit(
                    process_task_evolving, seed_pool, pool_lock, original_seed_ids, 
                    idx, api_key, task_seed, parent_questions
                )
                pending_futures.add(future)
                submitted += 1
            
            while pending_futures:
                done_futures = {f for f in pending_futures if f.done()}
                if not done_futures:
                    time.sleep(0.01)
                    continue
                
                for future in done_futures:
                    pending_futures.remove(future)
                    try:
                        result, status, (level, subject), gen_usage, val_usage, new_question, idx = future.result()
                        tag = f"L{level}/{subject}"
                        
                        for k in total_gen_usage:
                            total_gen_usage[k] += gen_usage.get(k, 0)
                            total_val_usage[k] += val_usage.get(k, 0)
                        
                        question_id = next_question_id[0]
                        next_question_id[0] += 1
                        result["id"] = question_id
                        results.append(result)
                        
                        if result.get("valid") or result.get("gt_assigned"):
                            completed += 1
                            gt_status = "GT" if result["ground_truth"] is not None else "no-GT"
                            print(f"  [OK {completed}/{total}] {tag} gen:{result['generator_idx']} val:{result['validator_idx']} ({gt_status})")
                            
                            if new_question:
                                with pool_lock:
                                    seed_pool[idx] = (level, subject, new_question)
                                    parent_questions[idx] = question_id
                        else:
                            failed += 1
                            failure_counts[status] = failure_counts.get(status, 0) + 1
                            print(f"  [FAIL {failed}/{total}] {tag}: {status}")
                    except Exception as e:
                        question_id = next_question_id[0]
                        next_question_id[0] += 1
                        failed_result = build_failed_result(None, None, None, None, "exception")
                        failed_result["id"] = question_id
                        results.append(failed_result)
                        failed += 1
                        failure_counts["exception"] = failure_counts.get("exception", 0) + 1
                        print(f"  [FAIL {failed}/{total}] {type(e).__name__}")
                    
                    if submitted < total:
                        idx = base_rng.randint(0, pool_size - 1)
                        task_seed = base_rng.randint(0, 2**31)
                        future = executor.submit(
                            process_task_evolving, seed_pool, pool_lock, original_seed_ids,
                            idx, api_key, task_seed, parent_questions
                        )
                        pending_futures.add(future)
                        submitted += 1
    
    # Assign IDs to results (for batch and static modes)
    if seed_mode != "evolving":
        for i, r in enumerate(results):
            r["id"] = i
    
    # Sort results (handle None questions)
    results.sort(key=lambda x: (x["level"] or 0, x["subject"] or "", x["question"] or ""))
    
    # Duplicate detection on questions with gt_assigned
    print(f"\nRunning duplicate detection (threshold={args.duplicate_threshold})...")
    gt_questions = [r for r in results if r.get("gt_assigned")]
    gt_indices = [i for i, r in enumerate(results) if r.get("gt_assigned")]
    
    num_duplicates = 0
    if gt_questions:
        pairs = find_similar_pairs(gt_questions, threshold=args.duplicate_threshold, verbose=True)
        clusters = get_duplicate_clusters(pairs, threshold=args.duplicate_threshold)
        
        for cluster in clusters:
            first_idx = cluster[0]
            first_id = gt_questions[first_idx]["id"]
            for dup_idx in cluster[1:]:
                results_idx = gt_indices[dup_idx]
                results[results_idx]["is_duplicate"] = True
                results[results_idx]["duplicate_of"] = first_id
                results[results_idx]["valid"] = False
                num_duplicates += 1
        
        print(f"  Found {len(clusters)} duplicate clusters, {num_duplicates} questions marked as duplicates")
    else:
        print("  No questions with ground truth to check for duplicates")
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    total_cost = end_usage - start_usage
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), "data", f"augmented_math_{timestamp}.json")
    
    original_seeds_output = [
        {"seed_id": s["seed_id"], "level": s["level"], "subject": s["subject"], "question": s["question"]}
        for s in original_seeds
    ]
    
    # Compute summary stats
    generation_ok_count = sum(1 for r in results if r.get("generation_ok"))
    gt_count = sum(1 for r in results if r.get("gt_assigned"))
    valid_count = sum(1 for r in results if r.get("valid"))
    
    output_data = {
        "metadata": {
            "generator_model": GENERATOR_MODEL,
            "validator_model": VALIDATOR_MODEL,
            "levels": levels,
            "subjects": subjects,
            "seed_mode": seed_mode,
            "num_questions_requested": args.num_questions,
            "total_seeds": total_seeds,
            "inflation_factor": inflation,
            "questions_per_seed": questions_per_seed if seed_mode == "batch" else None,
            "num_mc_options": NUM_MC_OPTIONS,
            "duplicate_threshold": args.duplicate_threshold,
            "timestamp": timestamp,
            "total_questions": len(results),
            "generation_ok_count": generation_ok_count,
            "gt_assigned_count": gt_count,
            "duplicate_count": num_duplicates,
            "valid_count": valid_count,
            "failure_counts": failure_counts,
            "generator_usage": total_gen_usage,
            "validator_usage": total_val_usage,
            "total_cost": total_cost
        },
        "original_seeds": original_seeds_output,
        "questions": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Calculate GT agreement rates
    gt_agreement = gt_count / generation_ok_count if generation_ok_count else 0
    gen_ok_non_dup = generation_ok_count - num_duplicates
    gt_non_dup = gt_count - num_duplicates
    gt_agreement_post_dedup = gt_non_dup / gen_ok_non_dup if gen_ok_non_dup else 0
    
    print(f"\n{'='*50}")
    print(f"Total questions: {len(results)}")
    print(f"  Generation OK: {generation_ok_count} ({100*generation_ok_count/len(results) if results else 0:.1f}%)")
    print(f"  GT assigned:   {gt_count} ({100*gt_count/len(results) if results else 0:.1f}%)")
    print(f"  GT agreement:  {gt_count}/{generation_ok_count} ({100*gt_agreement:.1f}%)")
    print(f"  Duplicates:    {num_duplicates}")
    print(f"  Valid:         {valid_count} ({100*valid_count/len(results) if results else 0:.1f}%)")
    print(f"  GT agreement (post-dedup): {gt_non_dup}/{gen_ok_non_dup} ({100*gt_agreement_post_dedup:.1f}%)")
    if failure_counts:
        print(f"Failure breakdown: {failure_counts}")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
