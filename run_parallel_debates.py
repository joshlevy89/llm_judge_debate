#!/usr/bin/env python3
"""
Run multiple debates in parallel.

This script orchestrates multiple instances of run_single_debate.py,
allowing you to run debates simultaneously with separate log files
and aggregated results.
"""

import os
import sys
import subprocess
import argparse
import time
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import glob

from config import MAX_TURNS_DEFAULT


def run_single_debate_process(debate_id: int, output_dir: str, jsonl_filename: str, seed: int = None, 
                               max_turns: int = MAX_TURNS_DEFAULT, quiet: bool = False, 
                               master_seed: int = None) -> subprocess.Popen:
    """
    Launch a single debate as a subprocess.
    
    Args:
        debate_id: Unique identifier for this debate instance
        output_dir: Directory for output files
        seed: Random seed for reproducibility
        max_turns: Maximum debate turns
        quiet: Suppress verbose output in the subprocess
        
    Returns:
        subprocess.Popen object
    """
    # Generate unique run identifier
    import uuid
    run_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        '-u',  # Unbuffered output for real-time logging
        'run_single_debate.py',
        '--output-dir', output_dir,
        '--jsonl-filename', jsonl_filename,
        '--max-turns', str(max_turns),
        '--run-id', run_id,  # Pass unique identifier
    ]
    
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    
    if master_seed is not None:
        cmd.extend(['--master-seed', str(master_seed)])
    
    if quiet:
        cmd.append('--quiet')
    
    # Launch process with output redirected to devnull (we save everything in detail.txt files)
    print(f"[run_id: {run_id}] Starting...")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    return process, run_id


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
    debater_correct = sum(1 for r in results if r.get('debater_direct', {}).get('correct'))
    judge_direct_correct = sum(1 for r in results if r.get('judge_direct', {}).get('correct'))
    
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    print(f"Total debates: {total}")
    print(f"Debater direct QA accuracy: {debater_correct}/{total} ({100*debater_correct/total:.1f}%)")
    print(f"Judge direct QA accuracy: {judge_direct_correct}/{total} ({100*judge_direct_correct/total:.1f}%)")
    
    if has_interactive:
        # Count results that have interactive mode
        interactive_results = [r for r in results if 'interactive' in r.get('modes', {})]
        interactive_correct = sum(1 for r in interactive_results if r['modes']['interactive'].get('correct'))
        print(f"Judge after interactive debate accuracy: {interactive_correct}/{len(interactive_results)} ({100*interactive_correct/len(interactive_results):.1f}%)")
    
    if has_non_interactive:
        # Count results that have non-interactive mode
        non_interactive_results = [r for r in results if 'non_interactive' in r.get('modes', {})]
        non_interactive_correct = sum(1 for r in non_interactive_results if r['modes']['non_interactive'].get('correct'))
        print(f"Judge after non-interactive debate accuracy: {non_interactive_correct}/{len(non_interactive_results)} ({100*non_interactive_correct/len(non_interactive_results):.1f}%)")
    
    print("="*70)


def print_debate_progress_hint():
    """Print helpful info about monitoring debates."""
    print("\n" + "="*70)
    print("MONITORING DEBATES")
    print("="*70)
    print("Debates are running in the background.")
    print("Results are saved to debate_detail_*.txt files as they complete.")
    print("Use --wait flag to wait for completion and see aggregate statistics.")
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (generates deterministic seeds for each debate)')
    parser.add_argument('--max-turns', type=int, default=MAX_TURNS_DEFAULT,
                        help=f'Maximum number of debate turns (default: {MAX_TURNS_DEFAULT})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output in debate processes')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for all debates to complete before exiting')
    parser.add_argument('--auto-monitor', action='store_true',
                        help='[DEPRECATED] No longer used - log files removed for efficiency')
    
    args = parser.parse_args()
    
    # Generate deterministic seeds for each debate if master seed provided
    debate_seeds = None
    if args.seed is not None:
        random.seed(args.seed)
        debate_seeds = [random.randint(1, 1000000) for _ in range(args.num_debates)]
    
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
    print("="*70)
    
    # Note: No need to backup individual JSONL since each run has its own folder
    
    # Launch debate processes in batches
    all_processes = []
    debates_to_run = list(range(args.num_debates))
    
    # Process debates in batches
    while debates_to_run:
        # Launch next batch
        batch = debates_to_run[:max_concurrent]
        debates_to_run = debates_to_run[max_concurrent:]
        
        batch_processes = []
        for i in batch:
            seed = debate_seeds[i] if debate_seeds else None
            process, run_id = run_single_debate_process(
                debate_id=i+1,
                output_dir=run_output_dir,
                jsonl_filename=Path(master_jsonl).name,  # Just the filename, not full path
                seed=seed,
                max_turns=args.max_turns,
                quiet=args.quiet,
                master_seed=args.seed
            )
            batch_processes.append((run_id, process))
        
        all_processes.extend(batch_processes)
        
        # If we have more batches to run, wait for this batch to complete
        if debates_to_run:
            print(f"\nWaiting for batch of {len(batch_processes)} debates to complete before starting next batch...")
            batch_completed = [False] * len(batch_processes)
            
            while not all(batch_completed):
                for idx, (run_id, process) in enumerate(batch_processes):
                    if not batch_completed[idx]:
                        retcode = process.poll()
                        if retcode is not None:
                            batch_completed[idx] = True
                            if retcode == 0:
                                print(f"[run_id: {run_id}] ✓ Completed")
                            else:
                                print(f"[run_id: {run_id}] ✗ Failed (exit code {retcode})")
                
                if not all(batch_completed):
                    time.sleep(1)
            
            print(f"Batch complete. {len(debates_to_run)} debates remaining.")
    
    processes = all_processes
    
    # Print monitoring hints
    print_debate_progress_hint()
    
    # Remove auto-monitor functionality since we no longer have log files
    if args.auto_monitor:
        print("\nNote: --auto-monitor flag is deprecated (log files removed for efficiency).")
        print("Results are saved to debate_detail_*.txt files as debates complete.")
    
    if not args.wait:
        print("\nDebates are running in the background.")
        print("\nTo wait for completion and see aggregated results, re-run with --wait flag")
        return
    
    # Wait for all processes to complete
    print("\nWaiting for all debates to complete...")
    print("(This may take several minutes)")
    print()
    
    completed = [False] * len(processes)
    
    while not all(completed):
        for idx, (run_id, process) in enumerate(processes):
            if not completed[idx]:
                retcode = process.poll()
                if retcode is not None:
                    completed[idx] = True
                    if retcode == 0:
                        print(f"[run_id: {run_id}] ✓ Completed successfully")
                    else:
                        print(f"[run_id: {run_id}] ✗ Failed with exit code {retcode}")
        
        if not all(completed):
            time.sleep(1)
    
    print("\n" + "="*70)
    print("All debates completed!")
    print("="*70)
    
    # Print statistics (debates already wrote directly to master JSONL)
    print_aggregate_stats(master_jsonl)
    
    # List generated files
    print("\nGenerated files:")
    print(f"  Results JSONL: {master_jsonl}")
    
    # List detail files
    detail_files = sorted(Path(run_output_dir).glob('debate_detail_*.txt'))
    if detail_files:
        print(f"  Detail files ({len(detail_files)}):")
        for detail_file in detail_files[-args.num_debates:]:  # Show only recent ones
            print(f"    {detail_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

