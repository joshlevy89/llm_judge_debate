"""
Evaluate generated questions with Gemini 3 Flash (validator) and Llama 3.1 8B.

Runs each model on the multiple choice questions and parses their answers.
Ground truth is established when both generator and validator agree.
"""

import os
import re
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_utils import call_openrouter, get_openrouter_key_info, parse_answer

# Load prompts
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
with open(os.path.join(PROMPTS_DIR, "qa_prompts.yaml")) as f:
    QA_PROMPTS = yaml.safe_load(f)
with open(os.path.join(PROMPTS_DIR, "shared_prompts.yaml")) as f:
    SHARED_PROMPTS = yaml.safe_load(f)

# Model configurations
MODELS = {
    "gemini_validator": {
        "id": "google/gemini-3-flash-preview",
        "reasoning_max_tokens": 5000,
        "max_tokens": 15000
    },
    "llama": {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "max_tokens": 20000
    }
}

# OLD PROMPT - commented out for testing structured prompts
# MC_PROMPT = """Answer the following multiple choice math problem.
#
# Problem:
# {question}
#
# Options:
# {options}
#
# Think through the problem step by step, then provide your final answer.
# Your final answer MUST be on its own line in exactly this format:
# ANSWER: <single digit 0-9>
#
# For example, if you think option 3 is correct, write:
# ANSWER: 3"""


def build_prompt(question, options_text):
    """Build prompt using structured templates from yaml files."""
    response_format = SHARED_PROMPTS["response_format_prompt"].replace("{N}", str(len(options_text.split("\n"))))
    return QA_PROMPTS["qa_prompt_template"].format(
        question=question,
        options_text=options_text,
        response_format_prompt=response_format
    )


def format_options(options):
    """Format options as numbered list."""
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))


# OLD PARSER - commented out for testing structured prompts
# def parse_mc_answer(response_text):
#     """Parse the model's MC answer from response."""
#     if not response_text:
#         return None
#     
#     # Look for "ANSWER: X" pattern
#     match = re.search(r"ANSWER:\s*(\d)", response_text, re.IGNORECASE)
#     if match:
#         return int(match.group(1))
#     
#     # Fallback: look for standalone digit at end
#     lines = response_text.strip().split("\n")
#     for line in reversed(lines):
#         line = line.strip()
#         if line.isdigit() and len(line) == 1:
#             return int(line)
#     
#     return None


def evaluate_question(question_data, model_name, model_config, api_key):
    """Run a single model on a question."""
    options_text = format_options(question_data["options"])
    prompt = build_prompt(question_data["question"], options_text)
    
    kwargs = {
        "max_tokens": model_config["max_tokens"]
    }
    if "reasoning_effort" in model_config:
        kwargs["reasoning_effort"] = model_config["reasoning_effort"]
    if "reasoning_max_tokens" in model_config:
        kwargs["reasoning_max_tokens"] = model_config["reasoning_max_tokens"]
    
    response, usage = call_openrouter(
        prompt, model_config["id"], api_key,
        temperature=0.5,
        **kwargs
    )
    
    content = response.get("content") or ""
    parsed = parse_answer(content, lenient=True)
    
    return {
        "model": model_name,
        "answer": parsed["answer"],
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"],
        "raw_content": content,
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
        "generator_answer_idx": question_data["correct_idx"],
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
    
    # Load questions and filter out invalid ones
    with open(args.questions_file) as f:
        all_questions = json.load(f)
    
    # Validate questions - must have required fields
    questions = []
    for q in all_questions:
        if (q.get("question") and 
            q.get("options") and 
            len(q["options"]) > 0 and
            q.get("correct_idx") is not None and
            0 <= q["correct_idx"] < len(q["options"])):
            questions.append(q)
    
    skipped = len(all_questions) - len(questions)
    print(f"Loaded {len(questions)} valid questions ({skipped} skipped as invalid)")
    
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
                
                validator_ans = result["evaluations"].get("gemini_validator", {}).get("answer")
                llama_ans = result["evaluations"].get("llama", {}).get("answer")
                generator_ans = result["generator_answer_idx"]
                
                print(f"[{completed}/{len(questions)}] Q{idx} - Generator:{generator_ans} Validator:{validator_ans} Llama:{llama_ans}")
                
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
    generator_validator_agree = sum(
        1 for r in results
        if r["generator_answer_idx"] == r["evaluations"].get("gemini_validator", {}).get("answer")
    )
    
    print(f"\n{'='*50}")
    print(f"Evaluated {len(results)} questions")
    print(f"Generator-Validator agreement: {generator_validator_agree}/{len(results)} ({100*generator_validator_agree/len(results):.1f}%)")
    print(f"Saved to: {output_path}")
    print(f"Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
