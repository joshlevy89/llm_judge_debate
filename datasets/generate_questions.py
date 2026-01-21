"""
Script 1: Generate augmented MATH questions using Gemini 3 Flash.

For each (level, subject) combination, generates 10 novel multiple-choice questions
by using seed questions from the MATH dataset as inspiration.
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

# Model configuration
MODEL = "google/gemini-3-flash-preview"
REASONING_MAX_TOKENS = 10000
MAX_TOKENS = 20000
TEMPERATURE = 0.7

# Dataset configuration
LEVELS = [1, 2, 3, 4, 5]
SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]
QUESTIONS_PER_CATEGORY = 10
MAX_SEED_ATTEMPTS = 5
NUM_MC_OPTIONS = 10
MAX_THREADS = 200

# Prompts
GENERATION_PROMPT = """Help the user to create a new math problem inspired by a given one. Make the new problem reasonable and solvable.

Here are some examples of how to complete this task.

Problem:
In the equation $$5x^2-kx+1=0$$ determine $k$ such that the difference of the roots be equal to unity.

Write another problem inspired by this one:
The roots $x_1$ and $x_2$ of the equation $$x^2-3ax+a^2=0$$ are such that $x_1^2+x_2^2=1.75$. Determine $a$.

Problem:
Solve the following equation $\\frac{{3+x}}{{3x}}=\\sqrt{{\\frac{{1}}{{9}}+\\frac{{1}}{{x}}\\sqrt{{\\frac{{4}}{{9}}+\\frac{{2}}{{x^2}}}}}}$

Write another problem inspired by this one:
Solve the following equation $\\sqrt{{1+x\\sqrt{{x^2+24}}}}=x+1$

Problem:
In an infinitely decreasing geometric progression the sum of all the terms occupying odd places is equal to 36, and that of all the terms at even places equals 12. Find the progression.

Write another problem inspired by this one:
The sum of the terms of an infinitely decreasing geometric progression is equal to 56, and the sum of the squared terms of the same progression is 448. Find the first term and the common ratio.

Problem:
Two railway stations are at a distance of 96 km from each other. One train covers this distance 40 minutes faster than does the other. The speed of the first train is 12 km/h higher than that of the second. Determine the speed of both trains.

Write another problem inspired by this one:
A student was asked to multiply 78 by a two-digit number in which the tens digit was three times as large as the units digit; by mistake, he interchanged the digits in the second factor and thus obtained a product smaller than the true product by 2808. What was the true product?

Here is the problem from the user:
{seed_question}

Write another problem inspired by this one.
Don't just change the numbers and context, but try to create a problem that requires another approach to solve.
Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
After the problem is generated finish your response right away."""

SOLVE_PROMPT = """Solve the following math problem. Show your work step by step, then provide the final answer.

Problem:
{question}

After solving, state your final answer clearly on its own line starting with "FINAL ANSWER: "."""

DISTRACTOR_PROMPT = """You are creating a multiple choice math question. Given the problem and its correct answer, generate {num_distractors} plausible but incorrect answer options.

Problem:
{question}

Correct Answer: {correct_answer}

Generate exactly {num_distractors} incorrect but plausible answers. These should be common mistakes a student might make. Output as a JSON array of strings.

Respond with ONLY the JSON array, nothing else. Example format:
["incorrect1", "incorrect2", "incorrect3", ...]"""


def load_math_dataset():
    """Load MATH dataset from HuggingFace."""
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


def generate_question(seed_question, api_key):
    """Generate a new question inspired by the seed."""
    prompt = GENERATION_PROMPT.format(seed_question=seed_question)
    try:
        response, usage = call_openrouter(
            prompt, MODEL, api_key,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_max_tokens=REASONING_MAX_TOKENS
        )
    except Exception as e:
        print(f"    [DEBUG] generate_question API error: {e}")
        raise
    
    content = response.get("content")
    if not content:
        print(f"    [DEBUG] generate_question empty content. Response: {response}")
    
    return content, {"prompt": prompt, "response": response, "usage": usage}


def solve_question(question, api_key):
    """Attempt to solve the generated question."""
    prompt = SOLVE_PROMPT.format(question=question)
    try:
        response, usage = call_openrouter(
            prompt, MODEL, api_key,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            reasoning_max_tokens=REASONING_MAX_TOKENS
        )
    except Exception as e:
        print(f"    [DEBUG] solve_question API error: {e}")
        raise
    
    content = response.get("content") or ""
    
    if not content:
        print(f"    [DEBUG] solve_question empty content. Response: {response}")
    
    # Extract final answer
    if "FINAL ANSWER:" in content:
        answer = content.split("FINAL ANSWER:")[-1].strip().split("\n")[0].strip()
        return answer, {"prompt": prompt, "response": response, "usage": usage}
    
    print(f"    [DEBUG] solve_question no FINAL ANSWER found. Content preview: {content[:200] if content else 'empty'}")
    return None, {"prompt": prompt, "response": response, "usage": usage}


def generate_distractors(question, correct_answer, api_key):
    """Generate plausible incorrect options."""
    prompt = DISTRACTOR_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        num_distractors=NUM_MC_OPTIONS - 1
    )
    try:
        response, usage = call_openrouter(
            prompt, MODEL, api_key,
            temperature=TEMPERATURE,
            max_tokens=2000,
            reasoning_max_tokens=2000
        )
    except Exception as e:
        print(f"    [DEBUG] generate_distractors API error: {e}")
        raise
    
    content = response.get("content") or ""
    
    if not content:
        print(f"    [DEBUG] generate_distractors empty content. Response: {response}")
        return None, {"prompt": prompt, "response": response, "usage": usage}
    
    try:
        # Parse JSON array from response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            distractors = json.loads(content[start:end])
            if len(distractors) >= NUM_MC_OPTIONS - 1:
                return distractors[:NUM_MC_OPTIONS - 1], {"prompt": prompt, "response": response, "usage": usage}
        print(f"    [DEBUG] generate_distractors couldn't find JSON array. Content: {content[:200]}")
    except json.JSONDecodeError as e:
        print(f"    [DEBUG] generate_distractors JSON parse error: {e}. Content: {content[:200]}")
    
    return None, {"prompt": prompt, "response": response, "usage": usage}


def create_mc_options(correct_answer, distractors, rng):
    """Shuffle correct answer randomly among distractors."""
    options = distractors + [correct_answer]
    rng.shuffle(options)
    correct_idx = options.index(correct_answer)
    return options, correct_idx


def generate_single_question(level, subject, seeds, api_key, question_num, rng_seed):
    """Generate a single question with MC options. Returns None on failure."""
    tag = f"L{level}/{subject} Q{question_num}"
    rng = random.Random(rng_seed)
    used_seeds = set()
    
    for attempt in range(MAX_SEED_ATTEMPTS):
        available = [s for s in seeds if s not in used_seeds]
        if not available:
            print(f"  [!] {tag}: No more seeds available")
            return None
        
        seed = rng.choice(available)
        used_seeds.add(seed)
        
        try:
            new_question, gen_raw = generate_question(seed, api_key)
            if not new_question:
                print(f"  [!] {tag}: Generation failed (attempt {attempt+1})")
                continue
            print(f"  [+] {tag}: Generated question (attempt {attempt+1})")
            
            answer, solve_raw = solve_question(new_question, api_key)
            if not answer:
                print(f"  [!] {tag}: Could not solve (attempt {attempt+1})")
                continue
            print(f"  [+] {tag}: Solved question (attempt {attempt+1})")
            
            distractors, dist_raw = generate_distractors(new_question, answer, api_key)
            if not distractors:
                print(f"  [!] {tag}: Distractor gen failed (attempt {attempt+1})")
                continue
            print(f"  [+] {tag}: Generated distractors (attempt {attempt+1})")
            
            options, correct_idx = create_mc_options(answer, distractors, rng)
            
            return {
                "level": level,
                "subject": subject,
                "question": new_question,
                "options": options,
                "correct_idx": correct_idx,
                "correct_answer": answer,
                "seed_question": seed,
                "raw": {
                    "generation": gen_raw,
                    "solving": solve_raw,
                    "distractors": dist_raw
                }
            }
            
        except Exception as e:
            print(f"  [!] {tag}: {type(e).__name__}: {e} (attempt {attempt+1})")
            continue
    
    print(f"  [X] {tag}: Failed after {MAX_SEED_ATTEMPTS} attempts")
    return None


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    math_data = load_math_dataset()
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
    # Build task list with deterministic seeds for each task
    tasks = []
    base_rng = random.Random(42)
    for level in LEVELS:
        for subject in SUBJECTS:
            seeds = math_data[level][subject]
            if len(seeds) < QUESTIONS_PER_CATEGORY:
                print(f"[!] Only {len(seeds)} seeds for L{level}/{subject}")
            for q_num in range(QUESTIONS_PER_CATEGORY):
                task_seed = base_rng.randint(0, 2**31)
                tasks.append((level, subject, seeds, q_num, task_seed))
    
    total = len(tasks)
    print(f"\nGenerating {total} questions with {MAX_THREADS} threads...")
    
    results = []
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(generate_single_question, level, subject, seeds, api_key, q_num, task_seed): (level, subject, q_num)
            for level, subject, seeds, q_num, task_seed in tasks
        }
        
        for future in as_completed(futures):
            level, subject, q_num = futures[future]
            tag = f"L{level}/{subject} Q{q_num}"
            try:
                result = future.result()
                if result:
                    results.append(result)
                    completed += 1
                    print(f"  [OK {completed}/{total}] {tag}")
                else:
                    failed += 1
                    print(f"  [FAIL {failed}] {tag}")
            except Exception as e:
                failed += 1
                print(f"  [FAIL {failed}] {tag}: {type(e).__name__}: {e}")
    
    # Sort by level, subject, question for consistent output
    results.sort(key=lambda x: (x["level"], x["subject"], x["question"]))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), "data", f"questions_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    cost = end_usage - start_usage
    
    print(f"\n{'='*50}")
    print(f"Generated {len(results)}/{total} questions ({failed} failed)")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
