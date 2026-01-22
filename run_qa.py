import os
import time
from pathlib import Path
from dotenv import load_dotenv
import config.config_qa as config_qa
from config.config_qa import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT, DATASET_FILTERS,
    MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    REASONING_EFFORT, REASONING_MAX_TOKENS, REASONING_ENABLED,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES,
    SPECIFIC_QUESTION_IDXS, MAX_THREADS, RERUN
)
from datasets import load_dataset
from utils.dataset_utils import select_questions_and_options
from utils.shared_utils import extract_config
from utils.qa_utils import run_qa_for_questions, get_existing_qa_keys, filter_existing_questions, load_specific_question_idxs

def main():
    load_dotenv()    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    results_path = Path('results') / 'qa' / 'qa_results.jsonl'
    dataset_config = extract_config(config_qa)
    
    print(f"Results will be appended to: {results_path}")
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    print(f"Using dataset filters {DATASET_FILTERS}")
    
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    
    dataset = dataset.add_column('_original_idx', range(len(dataset)))
    
    if DATASET_FILTERS:
        dataset = dataset.filter(lambda x: all(x.get(k) in v if isinstance(v, list) else x.get(k) == v for k, v in DATASET_FILTERS.items()))
        print(f"Filtered dataset to {len(dataset)} questions")

    if SPECIFIC_QUESTION_IDXS is not None:
        question_idxs = load_specific_question_idxs(SPECIFIC_QUESTION_IDXS)
        print(f"Using specific question indices (referring to original dataset): {len(question_idxs)} indices")
    else:
        print(f"Selecting {NUM_QUESTIONS} questions with seed {RANDOM_SEED}")
        import random
        rng = random.Random(RANDOM_SEED)
        filtered_idxs = rng.sample(range(len(dataset)), min(NUM_QUESTIONS, len(dataset)))
        question_idxs = [dataset[idx]['_original_idx'] for idx in filtered_idxs]
    
    if not RERUN:
        unfiltered_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
        existing_qa = get_existing_qa_keys(results_path)
        questions_data = select_questions_and_options(DATASET_NAME, unfiltered_dataset, len(question_idxs), NUM_CHOICES, None, question_idxs)
        question_idxs = filter_existing_questions(question_idxs, questions_data, MODEL_NAME, NUM_CHOICES, existing_qa, REASONING_EFFORT, REASONING_MAX_TOKENS)
        
        if not question_idxs:
            print("All questions already have QA results. Nothing to run. Can set RERUN = True to rerun.")
            return
        
        print(f"Filtered to {len(question_idxs)} questions without existing results")
    
    start_time = time.time()
    print(f"Processing {len(question_idxs)} questions...")
    
    result = run_qa_for_questions(
        question_idxs=question_idxs,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        reasoning_max_tokens=REASONING_MAX_TOKENS,
        reasoning_enabled=REASONING_ENABLED,
        dataset_config=dataset_config,
        num_choices=NUM_CHOICES,
        api_key=api_key,
        max_threads=MAX_THREADS,
        qa_results_path=results_path,
        random_seed=RANDOM_SEED
    )
    
    duration = time.time() - start_time
    
    print(f"\nRun ID: {result['run_id']}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: total {len(question_idxs)}, success {result['completed']}, error {result['failed']}")
    print(f"Cost: ${result['cost']:.6f}")

if __name__ == "__main__":
    main()
