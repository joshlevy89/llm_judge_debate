import json
import re

# filepath = "/Users/joshlevy/projects/llm_judge_debate/experiments/openmath_eval/results/openmath_eval_20260118_123416.json"
filepath = "/Users/joshlevy/projects/llm_judge_debate/experiments/augmented_math_results/augmented_math_20260120_142545.json"

def parse_answer(raw_response):
    patterns = [
        r'\\boxed\{.+s',
        r'\*\*[Aa]nswer:?\*\*\s*.+',
        r'[Aa]nswer\s*[:=]\s*.+',
        r'[Tt]he\s+final\s+answer\s+is:?\s*.+',
    ]
    
    last_match = None
    last_pos = -1
    
    for pattern in patterns:
        for match in re.finditer(pattern, raw_response, re.IGNORECASE):
            if match.start() > last_pos:
                last_pos = match.start()
                last_match = match.group(0)
    
    if last_match:
        return last_match.split('\n')[0].strip()
    return None

with open(filepath) as f:
    data = json.load(f)

for model_name, entries in data["results"].items():
    for entry in entries:
        # print(f"idx: {entry['idx']}")
        print(f"idx: {entry['id']}")
        # print(f"problem_source: {entry['problem_source']}")
        print(f"model: {entry['model']}")
        # print(f"expected: {entry['expected']}")
        raw_response = entry.get('raw_response')
        # print(f"raw_response: {raw_response}")
        print(f"raw_response: {raw_response}")
        parsed = parse_answer(raw_response) if raw_response else None
        if parsed and len(parsed) > 100:
            parsed = None
        print(f"Parsed Answer: {parsed}")
        print("-" * 80)
