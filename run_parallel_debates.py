#!/usr/bin/env python3
"""
Run multiple debates in parallel using threading.

This script orchestrates multiple debate instances using threading for efficient
I/O-bound parallel execution (LLM API calls).
"""

import os
import sys
import argparse
import time
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from config import MAX_TURNS_DEFAULT, MIN_TURNS_DEFAULT, MASTER_SEED, DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT
from run_single_debate import run_single_debate_logic


def run_single_debate_thread(debate_id: int, output_dir: str, jsonl_filename: str, question_idx: int = None, 
                             max_turns: int = MAX_TURNS_DEFAULT, min_turns: int = MIN_TURNS_DEFAULT,
                             quiet: bool = True, master_seed: int = None, dataset=None) -> Dict:
    """
    Run a single debate in a thread.
    
    Args:
        debate_id: Unique identifier for this debate instance
        output_dir: Directory for output files
        jsonl_filename: JSONL filename for results
        question_idx: Question index to use (None = random)
        max_turns: Maximum debate turns
        min_turns: Minimum debate turns before judge can end
        quiet: Suppress verbose output (default True for threading)
        master_seed: Master seed for logging
        dataset: Pre-loaded dataset to avoid concurrent HuggingFace requests
        
    Returns:
        Dict with run results
    """
    run_id = str(uuid.uuid4())[:8]
    
    print(f"[run_id: {run_id}] Starting...")
    
    try:
        result = run_single_debate_logic(
            output_dir=output_dir,
            question_idx=question_idx,
            master_seed=master_seed,
            max_turns=max_turns,
            min_turns=min_turns,
            quiet=quiet,
            run_id=run_id,
            jsonl_filename=jsonl_filename,
            dataset=dataset
        )
        print(f"[run_id: {run_id}] ✓ Completed")
        return result
    except Exception as e:
        print(f"[run_id: {run_id}] ✗ Failed: {e}")
        raise


def print_aggregate_stats(master_jsonl: str):
    """Print summary statistics from master JSONL file."""
    if not Path(master_jsonl).exists():
        return
    
    # Load all results from JSONL
    results = []
    with open(master_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return
    
    # Check which modes are available
    has_interactive = any('interactive' in r.get('modes', {}) for r in results)
    has_non_interactive = any('non_interactive' in r.get('modes', {}) for r in results)
    
    total = len(results)
    
    debater_direct_results = [r for r in results if r.get('debater_direct') is not None]
    debater_correct = sum(1 for r in debater_direct_results if r['debater_direct'].get('correct'))
    
    judge_direct_results = [r for r in results if r.get('judge_direct') is not None]
    judge_direct_correct = sum(1 for r in judge_direct_results if r['judge_direct'].get('correct'))
    
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    print(f"Total debates: {total}")
    print(f"Debater direct QA accuracy: {debater_correct}/{len(debater_direct_results)} ({100*debater_correct/len(debater_direct_results):.1f}%)")
    print(f"Judge direct QA accuracy: {judge_direct_correct}/{len(judge_direct_results)} ({100*judge_direct_correct/len(judge_direct_results):.1f}%)")
    
    if has_interactive:
        interactive_results = [r for r in results if r.get('modes', {}).get('interactive') is not None]
        interactive_correct = sum(1 for r in interactive_results if r['modes']['interactive'].get('correct'))
        print(f"Judge after interactive debate accuracy: {interactive_correct}/{len(interactive_results)} ({100*interactive_correct/len(interactive_results):.1f}%)")
    
    if has_non_interactive:
        non_interactive_results = [r for r in results if r.get('modes', {}).get('non_interactive') is not None]
        non_interactive_correct = sum(1 for r in non_interactive_results if r['modes']['non_interactive'].get('correct'))
        print(f"Judge after non-interactive debate accuracy: {non_interactive_correct}/{len(non_interactive_results)} ({100*non_interactive_correct/len(non_interactive_results):.1f}%)")
    
    print("="*70)


def print_debate_progress_hint():
    """Print helpful info about monitoring debates."""
    print("\n" + "="*70)
    print("MONITORING DEBATES")
    print("="*70)
    print("Debates are running via threading (efficient for I/O-bound LLM calls).")
    print("Results are saved to log_*.txt files as they complete.")
    print("The script will wait for all debates to finish.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple debates in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 2 debates in parallel
  python run_parallel_debates.py --num-debates 2
  # Creates: master_results_random_n2_20251016_120155/ with JSONL and detail files

  # Run 100 debates with max 20 concurrent (to avoid rate limits)
  python run_parallel_debates.py --num-debates 100 --max-concurrent 20 --wait
  # Runs in batches of 20 until all 100 are complete

  # Run with specific seed for reproducibility
  python run_parallel_debates.py --num-debates 100 --seed 42 --max-concurrent 20 --wait
  # Creates: master_results_seed42_n100_20251016_120155/
  # The seed 42 generates deterministic seeds for all 100 debates

  # Run with custom max-turns
  python run_parallel_debates.py --num-debates 10 --seed 100 --max-turns 10
  # Creates: master_results_seed100_n10_turns10_20251016_120155/

  # Run quietly (suppress verbose debate output)
  python run_parallel_debates.py --num-debates 2 --quiet
        """
    )
    
    parser.add_argument('--num-debates', type=int, default=2,
                        help='Number of debates to run in parallel (default: 2)')
    parser.add_argument('--max-concurrent', type=int, default=None,
                        help='Maximum number of debates to run concurrently (default: all at once). Use this to avoid rate limits.')
    parser.add_argument('--output-dir', type=str, default='./parallel_debate_runs',
                        help='Output directory for results (default: ./parallel_debate_runs)')
    parser.add_argument('--master-jsonl', type=str, default=None,
                        help='Path to master JSONL for aggregated results (default: auto-generated descriptive name)')
    parser.add_argument('--seed', type=int, default=MASTER_SEED,
                        help=f'Master seed - samples questions without replacement (default: {MASTER_SEED or "random"})')
    parser.add_argument('--max-turns', type=int, default=MAX_TURNS_DEFAULT,
                        help=f'Maximum number of debate turns (default: {MAX_TURNS_DEFAULT})')
    parser.add_argument('--min-turns', type=int, default=MIN_TURNS_DEFAULT,
                        help=f'Minimum number of debate turns before judge can end (default: {MIN_TURNS_DEFAULT})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output in debate processes')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for all debates to complete before exiting')
    parser.add_argument('--auto-monitor', action='store_true',
                        help='[DEPRECATED] No longer used - log files removed for efficiency')
    
    args = parser.parse_args()
    
    # Load dataset once upfront to avoid concurrent HuggingFace requests from all threads
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)
    dataset_size = len(dataset)
    print(f"Dataset loaded: {dataset_size} questions available")
    
    # Sample question indices without replacement using master seed
    question_indices = None
    if args.seed is not None:
        random.seed(args.seed)
        
        # Sample without replacement
        question_indices = random.sample(range(dataset_size), min(args.num_debates, dataset_size))
        
        if args.num_debates > dataset_size:
            print(f"WARNING: Only {dataset_size} questions available, running {dataset_size} debates instead of {args.num_debates}.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate descriptive run folder name and master JSONL filename
    if args.master_jsonl:
        master_jsonl = args.master_jsonl
        # Extract folder name from custom JSONL path
        run_folder_name = Path(master_jsonl).stem
        run_output_dir = str(Path(args.output_dir) / run_folder_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build descriptive filename
        filename_parts = ['run']
        
        # # Add seed info
        # if args.seed is not None:
        #     filename_parts.append(f'seed{args.seed}')
        # else:
        #     filename_parts.append('random')
        
        # Add number of debates
        # filename_parts.append(f'n{args.num_debates}')
        
        # Add max turns if not default
        # if args.max_turns != MAX_TURNS_DEFAULT:
        #     filename_parts.append(f'turns{args.max_turns}')
        
        # Add timestamp
        filename_parts.append(timestamp)
        
        run_folder_name = '_'.join(filename_parts)
        run_output_dir = str(Path(args.output_dir) / run_folder_name)
        master_jsonl = str(Path(run_output_dir) / 'master_results.jsonl')
    
    # Create run-specific output directory
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Determine concurrency limit
    max_concurrent = args.max_concurrent if args.max_concurrent else args.num_debates
    
    print("="*70)
    print("PARALLEL DEBATE RUNNER")
    print("="*70)
    print(f"Number of debates: {args.num_debates}")
    if args.max_concurrent and args.max_concurrent < args.num_debates:
        print(f"Max concurrent: {max_concurrent} (batched to avoid rate limits)")
    print(f"Run folder: {run_output_dir}")
    print(f"Master JSONL: {master_jsonl}")
    print(f"Max turns per debate: {args.max_turns}")
    print(f"Min turns per debate: {args.min_turns}")
    print("="*70)
    
    # Note: No need to backup individual JSONL since each run has its own folder
    
    # Print monitoring hints
    print_debate_progress_hint()
    
    # Remove auto-monitor functionality
    if args.auto_monitor:
        print("\nNote: --auto-monitor flag is deprecated.")
        print("Results are saved to log_*.txt files as debates complete.")
    
    # Run debates using ThreadPoolExecutor
    print("\nStarting debates...")
    print("(This may take several minutes)")
    print()
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all debate tasks
        futures = []
        for i in range(args.num_debates):
            question_idx = question_indices[i] if question_indices else None
            future = executor.submit(
                run_single_debate_thread,
                debate_id=i+1,
                output_dir=run_output_dir,
                jsonl_filename=Path(master_jsonl).name,
                question_idx=question_idx,
                max_turns=args.max_turns,
                min_turns=args.min_turns,
                quiet=args.quiet,
                master_seed=args.seed,
                dataset=dataset
            )
            futures.append(future)
        
        if not args.wait:
            print("\nDebates are running in the background (via threads).")
            print("\nNote: With threading, debates run in the same process.")
            print("You can monitor progress by checking log_*.txt files.")
            # Note: Can't return early with threading as easily, so we wait
        
        # Wait for all debates to complete
        completed_count = 0
        failed_count = 0
        
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Debate failed with error: {e}")
        
        print(f"\nCompleted: {completed_count}, Failed: {failed_count}")
    
    print("\n" + "="*70)
    print("All debates completed!")
    print("="*70)
    
    # Print statistics (debates already wrote directly to master JSONL)
    print_aggregate_stats(master_jsonl)
    
    # List generated files
    print("\nGenerated files:")
    print(f"  Results JSONL: {master_jsonl}")
    
    # List log files
    log_files = sorted(Path(run_output_dir).glob('log_*.txt'))
    if log_files:
        print(f"  Log files ({len(log_files)}):")
        for log_file in log_files[-min(10, args.num_debates):]:  # Show up to 10 recent ones
            print(f"    {log_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

