"""
Script 2: Evaluate generated questions with Grok 4.1 Fast and Llama 3.1 8B.

Runs each model on the multiple choice questions and parses their answers.
"""

import os
import re
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import call_openrouter, get_openrouter_key_info

# Model configurations
MODELS = {
    "grok": {
        "id": "x-ai/grok-4.1-fast",
        "reasoning_effort": "medium",
        "max_tokens": 20000
    },
    "llama": {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "max_tokens": 20000
    }
}

MC_PROMPT = """Answer the following multiple choice math problem.

Problem:
{question}

Options:
{options}

Think through the problem step by step, then provide your final answer.
Your final answer MUST be on its own line in exactly this format:
ANSWER: <single digit 0-9>

For example, if you think option 3 is correct, write:
ANSWER: 3"""


def format_options(options):
    """Format options as numbered list."""
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))


def parse_mc_answer(response_text):
    """Parse the model's MC answer from response."""
    if not response_text:
        return None
    
    # Look for "ANSWER: X" pattern
    match = re.search(r"ANSWER:\s*(\d)", response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback: look for standalone digit at end
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.isdigit() and len(line) == 1:
            return int(line)
    
    return None


def evaluate_question(question_data, model_name, model_config, api_key):
    """Run a single model on a question."""
    prompt = MC_PROMPT.format(
        question=question_data["question"],
        options=format_options(question_data["options"])
    )
    
    kwargs = {
        "max_tokens": model_config["max_tokens"]
    }
    if "reasoning_effort" in model_config:
        kwargs["reasoning_effort"] = model_config["reasoning_effort"]
    
    response, usage = call_openrouter(
        prompt, model_config["id"], api_key,
        temperature=0.5,
        **kwargs
    )
    
    answer = parse_mc_answer(response.get("content"))
    
    return {
        "model": model_name,
        "answer": answer,
        "raw_content": response.get("content"),
        "usage": usage
    }


def process_question(idx, question_data, api_key):
    """Evaluate a question with all models."""
    evaluations = {}
    
    for model_name, model_config in MODELS.items():
        try:
            result = evaluate_question(question_data, model_name, model_config, api_key)
            evaluations[model_name] = result
        except Exception as e:
            evaluations[model_name] = {
                "model": model_name,
                "answer": None,
                "error": str(e)
            }
    
    return {
        "idx": idx,
        "level": question_data["level"],
        "subject": question_data["subject"],
        "question": question_data["question"],
        "options": question_data["options"],
        "gemini_answer_idx": question_data["correct_idx"],
        "evaluations": evaluations
    }


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluate questions with Grok and Llama")
    parser.add_argument("questions_file", help="Path to questions JSON file")
    parser.add_argument("--max-threads", type=int, default=200, help="Max parallel threads")
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    # Load questions
    with open(args.questions_file) as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Track cost
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
    # Evaluate
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {
            executor.submit(process_question, idx, q, api_key): idx
            for idx, q in enumerate(questions)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                grok_ans = result["evaluations"].get("grok", {}).get("answer")
                llama_ans = result["evaluations"].get("llama", {}).get("answer")
                gemini_ans = result["gemini_answer_idx"]
                
                print(f"[{completed}/{len(questions)}] Q{idx} - Gemini:{gemini_ans} Grok:{grok_ans} Llama:{llama_ans}")
                
            except Exception as e:
                print(f"[!] Q{idx} failed: {e}")
    
    # Sort by original index
    results.sort(key=lambda x: x["idx"])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(args.questions_file).replace(".json", "")
    output_path = os.path.join(os.path.dirname(args.questions_file), f"evaluations_{base_name}_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Report
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    cost = end_usage - start_usage
    
    # Quick agreement stats
    gemini_grok_agree = sum(
        1 for r in results
        if r["gemini_answer_idx"] == r["evaluations"].get("grok", {}).get("answer")
    )
    
    print(f"\n{'='*50}")
    print(f"Evaluated {len(results)} questions")
    print(f"Gemini-Grok agreement: {gemini_grok_agree}/{len(results)} ({100*gemini_grok_agree/len(results):.1f}%)")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
