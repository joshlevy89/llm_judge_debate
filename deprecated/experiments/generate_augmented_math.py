#!/usr/bin/env python3
"""
Generate augmented math questions using Claude Opus 4.5.
Pulls questions from the MATH dataset and creates inspired variations.
"""
import os
import json
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_utils import call_openrouter

SUBJECTS = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
]
LEVELS = [1, 2, 3, 4, 5]
QUESTIONS_PER_LEVEL = 10

MODEL = "anthropic/claude-opus-4"
TEMPERATURE = 0.7
MAX_TOKENS = 2000
MAX_THREADS = 20
SEED = 42

PROMPT_TEMPLATE = """Help the user to create a new math problem inspired by a given one. Make the new problem reasonable and solvable.

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
{question}

Write another problem inspired by this one.
Don't just change the numbers and context, but try to create a problem that requires another approach to solve.
Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
After the problem is generated finish your response right away."""


def load_math_questions():
    """Load and sample questions from the MATH dataset."""
    rng = random.Random(SEED)
    questions = []
    
    for subject in SUBJECTS:
        print(f"Loading {subject}...")
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        
        for level in LEVELS:
            filtered = [
                {"problem": item["problem"], "subject": subject, "level": level}
                for item in dataset
                if item["level"] == f"Level {level}"
            ]
            
            sampled = rng.sample(filtered, min(QUESTIONS_PER_LEVEL, len(filtered)))
            for i, q in enumerate(sampled):
                q["index"] = len(questions)
                q["subject_level_index"] = i
                questions.append(q)
    
    return questions


def generate_augmented(question, api_key):
    """Generate an augmented question using Opus 4.5."""
    prompt = PROMPT_TEMPLATE.format(question=question["problem"])
    
    response, usage = call_openrouter(
        prompt, MODEL, api_key, TEMPERATURE,
        max_tokens=MAX_TOKENS,
        run_id="augmented_math_gen",
        record_id=f"{question['subject']}_L{question['level']}_{question['subject_level_index']}",
        context="AugmentedMathGen"
    )
    
    return {
        "index": question["index"],
        "subject": question["subject"],
        "level": question["level"],
        "original_problem": question["problem"],
        "augmented_problem": response["content"],
        "token_usage": usage
    }


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    print("Loading MATH dataset...")
    questions = load_math_questions()
    print(f"Loaded {len(questions)} questions ({len(SUBJECTS)} subjects x {len(LEVELS)} levels x {QUESTIONS_PER_LEVEL} each)")
    
    results = []
    completed = [0]
    total = len(questions)
    
    print(f"\nGenerating augmented questions with {MODEL}...")
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(generate_augmented, q, api_key): q for q in questions}
        
        for future in as_completed(futures):
            q = futures[future]
            completed[0] += 1
            try:
                result = future.result()
                results.append(result)
                print(f"[{completed[0]}/{total}] {q['subject']} L{q['level']}")
            except Exception as e:
                print(f"[{completed[0]}/{total}] {q['subject']} L{q['level']} ERROR: {e}")
                results.append({
                    "index": q["index"],
                    "subject": q["subject"],
                    "level": q["level"],
                    "original_problem": q["problem"],
                    "augmented_problem": None,
                    "error": str(e)
                })
    
    results.sort(key=lambda x: x["index"])
    
    output_file = Path(__file__).parent / f"augmented_math_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "subjects": SUBJECTS,
                "levels": LEVELS,
                "questions_per_level": QUESTIONS_PER_LEVEL,
                "total_questions": len(results),
                "generated_at": datetime.now().isoformat()
            },
            "questions": results
        }, f, indent=2)
    
    successful = sum(1 for r in results if r.get("augmented_problem"))
    print(f"\nDone! {successful}/{len(results)} successful")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
