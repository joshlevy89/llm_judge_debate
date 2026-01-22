"""
Evaluate candidate validator models to determine if expensive reasoning tokens are necessary.

Compares multiple validator configurations:
- Gemini 3 Flash with 5k reasoning tokens (current validator)
- Gemini 3 Flash without reasoning
- DeepSeek v3.2 with reasoning enabled
- DeepSeek v3.2 without reasoning
"""

import os
import json
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

CANDIDATE_MODELS = {
    "gemini_thinking": {
        "id": "google/gemini-3-flash-preview",
        "reasoning_max_tokens": 5000,
        "max_tokens": 15000
    },
    "gemini_no_thinking": {
        "id": "google/gemini-3-flash-preview",
        "max_tokens": 15000
    },
    "deepseek_thinking": {
        "id": "deepseek/deepseek-v3.2",
        "reasoning_enabled": True,
        "max_tokens": 15000
    },
    "deepseek_no_thinking": {
        "id": "deepseek/deepseek-v3.2",
        "reasoning_enabled": False,
        "max_tokens": 15000
    }
}


def build_prompt(question, options_text):
    response_format = SHARED_PROMPTS["response_format_prompt"].replace("{N}", str(len(options_text.split("\n"))))
    return QA_PROMPTS["qa_prompt_template"].format(
        question=question,
        options_text=options_text,
        response_format_prompt=response_format
    )


def format_options(options):
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))


def evaluate_question(question_data, model_name, model_config, api_key):
    options_text = format_options(question_data["options"])
    prompt = build_prompt(question_data["question"], options_text)
    
    kwargs = {"max_tokens": model_config["max_tokens"]}
    if "reasoning_max_tokens" in model_config:
        kwargs["reasoning_max_tokens"] = model_config["reasoning_max_tokens"]
    if "reasoning_enabled" in model_config:
        kwargs["reasoning_enabled"] = model_config["reasoning_enabled"]
    
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
    evaluations = {}
    
    for model_name, model_config in CANDIDATE_MODELS.items():
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
    
    parser = argparse.ArgumentParser(description="Evaluate candidate validator models")
    parser.add_argument("questions_file", help="Path to questions JSON file")
    parser.add_argument("--max-threads", type=int, default=50, help="Max parallel threads")
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    
    with open(args.questions_file) as f:
        all_questions = json.load(f)
    
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
    print(f"Testing {len(CANDIDATE_MODELS)} model configurations: {list(CANDIDATE_MODELS.keys())}")
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get("data", {}).get("usage", 0) if key_info_start else 0
    
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
                
                answers = {name: result["evaluations"].get(name, {}).get("answer") 
                          for name in CANDIDATE_MODELS.keys()}
                gen_ans = result["generator_answer_idx"]
                
                print(f"[{completed}/{len(questions)}] Q{idx} - Gen:{gen_ans} " + 
                      " ".join(f"{k}:{v}" for k, v in answers.items()))
                
            except Exception as e:
                print(f"[!] Q{idx} failed: {e}")
    
    results.sort(key=lambda x: x["idx"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(args.questions_file).replace(".json", "")
    output_path = os.path.join(os.path.dirname(args.questions_file), 
                               f"candidate_validators_{base_name}_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get("data", {}).get("usage", 0) if key_info_end else 0
    cost = end_usage - start_usage
    
    print(f"\n{'='*60}")
    print(f"Evaluated {len(results)} questions with {len(CANDIDATE_MODELS)} models")
    
    for model_name in CANDIDATE_MODELS.keys():
        agree_with_gen = sum(
            1 for r in results
            if r["generator_answer_idx"] == r["evaluations"].get(model_name, {}).get("answer")
        )
        print(f"{model_name} agrees with generator: {agree_with_gen}/{len(results)} ({100*agree_with_gen/len(results):.1f}%)")
    
    print(f"\nSaved to: {output_path}")
    print(f"Total cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
