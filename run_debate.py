import os
import re
import json
from pandas.core.strings.accessor import NoNewAttributesMixin
import yaml
import random
import string
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import config.config_debate as config_debate
from config.config_debate import *
from utils.llm_utils import call_openrouter, get_openrouter_key_info, log_progress
from utils.dataset_utils import select_questions_and_options, format_options, load_dataset_unified
from utils.debate_utils import *
from utils.shared_utils import extract_config, generate_run_id, load_prompts


def setup_output_path(run_id):
    output_dir = Path('results') / 'debates'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{run_id}.jsonl'

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path(run_id)
    config = extract_config(config_debate)
    
    print(f"Run ID: {run_id}")
    print(f"Datetime: {run_datetime}")
    print(f"Results: {results_path}")
    
    if DATASET_NAME == "local":
        print(f"Loading local dataset: {DATASET_PATH}")
    else:
        print(f"Loading dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    
    dataset, metadata = load_dataset_unified(DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT, DATASET_PATH)
    
    if DATASET_NAME != "local":
        dataset = dataset.add_column('_original_idx', range(len(dataset)))
    
    if DATASET_FILTERS:
        dataset = dataset.filter(lambda x: all(x.get(k) in v if isinstance(v, list) else x.get(k) == v for k, v in DATASET_FILTERS.items()))
        print(f"Filtered dataset to {len(dataset)} questions")
    
    if SPECIFIC_IDXS is not None:
        question_idxs = load_specific_idxs(SPECIFIC_IDXS)
        print(f"Using specific question indices (referring to original dataset): {len(question_idxs)} indices")
    else:
        print(f"Selecting {NUM_QUESTIONS} questions with seed {RANDOM_SEED}")
        rng = random.Random(RANDOM_SEED)
        filtered_idxs = rng.sample(range(len(dataset)), min(NUM_QUESTIONS, len(dataset)))
        if DATASET_NAME == "local":
            question_idxs = [dataset[idx].get('idx', idx) for idx in filtered_idxs]
        else:
            question_idxs = [dataset[idx]['_original_idx'] for idx in filtered_idxs]
    
    if DATASET_NAME == "local":
        unfiltered_dataset, _ = load_dataset_unified(DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT, DATASET_PATH)
    else:
        from datasets import load_dataset
        unfiltered_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    questions_data = select_questions_and_options(DATASET_NAME, unfiltered_dataset, len(question_idxs), NUM_CHOICES, None, specific_idxs=question_idxs)
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    start_time = time.time()
    print(f"Processing {len(questions_data)} questions...")
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(process_question, q_data, INTERACTIVE_JUDGE, api_key, config, run_id, run_datetime): q_data
            for i, q_data in enumerate(questions_data)
        }
        
        with open(results_path, 'w') as f:
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    log_progress("completed", completed, len(questions_data), result['run_id'], result['record_id'], api_key, start_usage)
                    completed += 1
                else:
                    log_progress("failed", failed, len(questions_data), result['run_id'], result['record_id'], api_key, start_usage, error=result['error_message'])
                    failed += 1

                f.write(json.dumps(result) + '\n')
                f.flush()
    
    duration = time.time() - start_time
    
    print(f"\nRun ID: {run_id}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: total {len(questions_data)}, success {completed}, error {failed}")
    print(f"Model: {DEBATER_MODEL}")
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0)
    print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")

if __name__ == "__main__":
    main()

