from typing import Any


import os
import json
import yaml
import random
import string
import time
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import config.config_verdict as config_verdict
from config.config_verdict import (
    DEBATE_RUN_ID as CONFIG_DEBATE_RUN_ID, 
    JUDGE_MODEL as CONFIG_JUDGE_MODEL, 
    JUDGE_TEMPERATURE, JUDGE_REASONING_EFFORT, JUDGE_REASONING_MAX_TOKENS, JUDGE_REASONING_ENABLED,
    MAX_OUTPUT_TOKENS, SKIP_QA, RERUN, SUBSET_N, SPECIFIC_RECORD_IDS, 
    MAX_THREADS, UPTO_TURNS, SKIP_DEBATER_QA
)
from utils.llm_utils import call_openrouter, get_openrouter_key_info, parse_answer, log_progress
from utils.debate_utils import format_debate_history
from utils.shared_utils import extract_config, generate_run_id, load_prompts
from utils.qa_utils import format_qa_prompt, get_existing_qa_keys, run_qa_for_questions, normalize_whitespace

def setup_output_path(verdict_run_id):
    output_dir = Path('results') / 'verdicts'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{verdict_run_id}.jsonl'
    
def run_judge(question, options, public_debate_history_text, judge_template, response_format_prompt, api_key, verdict_run_id, record_id, judge_model):
    options_text = ""
    for i, option in enumerate[Any](options):
        options_text += f"Debater {i} is arguing for: {option}\n"
    
    min_confidence = int(100 / len(options))
    
    prompt = judge_template.format(
        question=question,
        num_debaters=len(options),
        options_text=options_text.strip(),
        public_debate_history_text=public_debate_history_text,
        min_confidence=min_confidence,
        response_format_prompt=response_format_prompt
    )
    
    response, token_usage = call_openrouter(
        prompt, 
        judge_model, 
        api_key, 
        JUDGE_TEMPERATURE,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        reasoning_max_tokens=JUDGE_REASONING_MAX_TOKENS,
        reasoning_enabled=JUDGE_REASONING_ENABLED,
        max_tokens=MAX_OUTPUT_TOKENS,
        run_id=verdict_run_id,
        record_id=record_id,
        context="Judge"
    )
    
    response_text = response['content']
    parsed = parse_answer(response_text)
    
    return {
        'raw_response': response_text,
        'internal_model_reasoning': response.get('reasoning'),
        'internal_model_reasoning_details': response.get('reasoning_details'),
        'parsed': parsed,
        'prompt': prompt,
        'token_usage': token_usage
    }

def check_and_run_missing_qa(debate_records, api_key, judge_model, max_threads):
    if SKIP_QA:
        return
    
    qa_results_path = Path('results') / 'qa' / 'qa_results.jsonl'
    existing_qa = get_existing_qa_keys(qa_results_path) if not RERUN else set()
    
    datasets = {}
    for record in debate_records:
        config = record['config']
        dataset_key = (config['dataset_name'], config['dataset_subset'], config['dataset_split'])
        if dataset_key not in datasets:
            datasets[dataset_key] = {
                'dataset_config': {
                    'dataset_name': config['dataset_name'],
                    'dataset_subset': config['dataset_subset'],
                    'dataset_split': config['dataset_split']
                },
                'debater_model': config['debater_model'],
                'debater_temperature': config['debater_temperature'],
                'random_seed': config.get('random_seed'),
                'records': []
            }
        datasets[dataset_key]['records'].append(record)
    
    for dataset_key, dataset_info in datasets.items():
        dataset_config = dataset_info['dataset_config']
        debater_model = dataset_info['debater_model']
        debater_temperature = dataset_info['debater_temperature']
        random_seed = dataset_info['random_seed']
        records = dataset_info['records']
        num_choices = len(records[0]['options'])
        
        print(f"\nProcessing dataset: {dataset_key[0]}/{dataset_key[1]} ({len(records)} records)")

        if judge_model == debater_model or SKIP_DEBATER_QA:
            tups = [(judge_model, JUDGE_TEMPERATURE)]
            if judge_model == debater_model:
                print("Since judge and debater are same, only running for one.")
            else:
                print('Since SKIP_DEBATER_QA is true, only running QA for judge.')
        else:
            tups = [(judge_model, JUDGE_TEMPERATURE), (debater_model, debater_temperature)]

        
        for model_name, temperature in tups:
            missing_question_idxs = []
            
            for record in records:
                question_idx = record['question_idx']
                num_choices = len(record['options'])
                prompt = format_qa_prompt(record['question'], record['options'], num_choices)
                
                key = (question_idx, model_name, normalize_whitespace(prompt))
                if key not in existing_qa:
                    if question_idx not in missing_question_idxs:
                        missing_question_idxs.append(question_idx)
            
            if missing_question_idxs:
                print(f"Found {len(missing_question_idxs)} questions without QA results for {model_name}. Running QA...")
                
                qa_result = run_qa_for_questions(
                    question_idxs=missing_question_idxs,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    reasoning_effort=JUDGE_REASONING_EFFORT,
                    reasoning_max_tokens=JUDGE_REASONING_MAX_TOKENS,
                    dataset_config=dataset_config,
                    num_choices=num_choices,
                    api_key=api_key,
                    max_threads=max_threads,
                    qa_results_path=qa_results_path,
                    random_seed=random_seed
                )
                
                print(f"QA completed: {qa_result['completed']} success, {qa_result['failed']} failed, cost ${qa_result['cost']:.6f}")
            else:
                print(f"All questions already have QA results for {model_name}. Skipping QA...")

def process_debate_record(debate_record, judge_template, response_format_prompt, judge_template_str, api_key, config, verdict_run_id, run_datetime, judge_model, debate_run_id, upto_turns):
    public_debate_history_text = format_debate_history(debate_record['debate_history'], show_private=False, upto_turns=upto_turns)
    
    judge_verdict = run_judge(
        debate_record['question'],
        debate_record['options'],
        public_debate_history_text,
        judge_template,
        response_format_prompt,
        api_key,
        verdict_run_id,
        debate_record['record_id'],
        judge_model
    )
    
    return {
        'verdict_run_id': verdict_run_id,
        'debate_run_id': debate_run_id,
        'record_id': debate_record['record_id'],
        'datetime': run_datetime,
        'config': config,
        'prompt_template': judge_template_str,
        'question': debate_record['question'],
        'options': debate_record['options'],
        'correct_idx': debate_record['correct_idx'],
        'success': True,
        'error_message': None,
        'judge_verdict': judge_verdict
    }

def main(judge_model=None, debate_run_id=None, upto_turns=None, max_threads=None):
    judge_model = judge_model or CONFIG_JUDGE_MODEL
    debate_run_id = debate_run_id or CONFIG_DEBATE_RUN_ID
    upto_turns = upto_turns if upto_turns is not None else UPTO_TURNS
    max_threads = max_threads or MAX_THREADS
    
    if judge_model is None:
        raise ValueError("No judge_model specified (set in config or pass as argument)")
    if debate_run_id is None:
        raise ValueError("No debate_run_id specified (set in config or pass as argument)")
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    verdict_run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path(verdict_run_id)
    config = extract_config(config_verdict)
    config['judge_model'] = judge_model
    config['debate_run_id'] = debate_run_id
    config['upto_turns'] = upto_turns
    
    print(f"Verdict Run ID: {verdict_run_id}")
    print(f"Debate Run ID: {debate_run_id}")
    print(f"Datetime: {run_datetime}")
    print(f"Results: {results_path}")
    print(f"Judge Model: {judge_model}")
    print(f"Upto Turns: {upto_turns}")
    
    if debate_run_id == 'human':
        debate_path = Path('results') / 'human' / f'human_interactive_debate.jsonl'
    else:
        debate_path = Path('results') / 'debates' / f'{debate_run_id}.jsonl'
    if not debate_path.exists():
        raise ValueError(f"Debate results not found: {debate_path}")
    
    debate_records = []
    with open(debate_path, 'r') as f:
        for line in f:
            debate_records.append(json.loads(line))
    
    if SPECIFIC_RECORD_IDS is not None:
        record_id_set = set(SPECIFIC_RECORD_IDS)
        debate_records = [r for r in debate_records if r['record_id'] in record_id_set]
    elif SUBSET_N is not None:
        debate_records = debate_records[:SUBSET_N]
    
    check_and_run_missing_qa(debate_records, api_key, judge_model, max_threads)
    
    judge_template = load_prompts('judge')
    response_format_prompt = load_prompts('shared')
    
    key_info_start = get_openrouter_key_info(api_key)
    start_usage = key_info_start.get('data', {}).get('usage', 0) if key_info_start else 0
    
    start_time = time.time()
    print(f"Processing {len(debate_records)} debates...")
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(process_debate_record, debate_record, judge_template, response_format_prompt, judge_template, api_key, config, verdict_run_id, run_datetime, judge_model, debate_run_id, upto_turns): debate_record
            for i, debate_record in enumerate(debate_records)
        }
        
        with open(results_path, 'w') as f:
            for future in as_completed(futures):
                debate_record = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    is_correct = result['judge_verdict']['parsed']['answer'] == result['correct_idx'] if result['judge_verdict']['parsed']['answer'] is not None else None
                    log_progress(f"{judge_model} completed", completed, len(debate_records), result['verdict_run_id'], result['record_id'], api_key, start_usage, is_correct=is_correct)
                except Exception as e:
                    failed += 1
                    error_trace = traceback.format_exc()
                    error_record_id = generate_run_id()
                    result = {
                        'success': False,
                        'error_message': error_trace,
                        'verdict_run_id': verdict_run_id,
                        'record_id': error_record_id,
                        'debate_run_id': debate_run_id,
                        'datetime': run_datetime,
                        'config': config,
                        'question': debate_record['question'],
                        'options': debate_record['options'],
                        'correct_idx': debate_record['correct_idx']
                    }
                    log_progress(f"{judge_model} failed", failed, len(debate_records), verdict_run_id, error_record_id, api_key, start_usage, error=error_trace)
                
                f.write(json.dumps(result) + '\n')
                f.flush()
    
    duration = time.time() - start_time
    
    results_path_str = f"results/verdicts/{verdict_run_id}.jsonl"
    with open(results_path_str, 'r') as f:
        results = [json.loads(line) for line in f]
    null_count = sum(1 for r in results if r.get('success') and r.get('judge_verdict', {}).get('parsed', {}).get('answer') is None)
    
    print(f"\nRun ID: {verdict_run_id}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: total {len(debate_records)}, success {completed}, error {failed}, null {null_count}")
    
    key_info_end = get_openrouter_key_info(api_key)
    end_usage = key_info_end.get('data', {}).get('usage', 0) if key_info_end else 0
    print(f"Cost: ${end_usage - start_usage:.6f} (Total: ${end_usage:.2f})")
    
    return verdict_run_id

if __name__ == "__main__":
    main()

