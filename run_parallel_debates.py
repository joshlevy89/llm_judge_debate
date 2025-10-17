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
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import glob


def run_single_debate_process(debate_id: int, output_dir: str, seed: int = None, 
                               max_turns: int = 20, quiet: bool = False) -> subprocess.Popen:
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
    
    # Create log file for this debate with unique identifier
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    seed_str = f'seed{seed}_' if seed is not None else ''
    log_file = Path(output_dir) / f'debate_{debate_id}_{seed_str}{run_id}_{timestamp}.log'
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        '-u',  # Unbuffered output for real-time logging
        'run_single_debate.py',
        '--output-dir', output_dir,
        '--max-turns', str(max_turns),
        '--run-id', run_id,  # Pass unique identifier
    ]
    
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    
    if quiet:
        cmd.append('--quiet')
    
    # Launch process with output redirected to log file
    print(f"[Debate {debate_id}] Starting... (run_id: {run_id}, log: {log_file})")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
    
    return process, log_file


def aggregate_results(output_dir: str, master_csv: str = None) -> str:
    """
    Aggregate all individual debate CSV results into a master CSV.
    
    Args:
        output_dir: Directory containing debate_results_summary.csv
        master_csv: Path to master CSV file (optional)
        
    Returns:
        Path to master CSV file
    """
    if master_csv is None:
        master_csv = Path(output_dir) / 'master_results_summary.csv'
    
    source_csv = Path(output_dir) / 'debate_results_summary.csv'
    
    if not source_csv.exists():
        print(f"Warning: No results file found at {source_csv}")
        return None
    
    # Read all rows from source CSV
    with open(source_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    if not rows:
        print("Warning: No results to aggregate")
        return None
    
    # Write to master CSV (append mode)
    file_exists = Path(master_csv).exists()
    
    with open(master_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nAggregated {len(rows)} result(s) to {master_csv}")
    return master_csv


def print_aggregate_stats(master_csv: str):
    """Print summary statistics from master CSV."""
    if not Path(master_csv).exists():
        return
    
    with open(master_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return
    
    total = len(rows)
    debater_correct = sum(1 for r in rows if r.get('debater_direct_correct') == 'True')
    judge_direct_correct = sum(1 for r in rows if r.get('judge_direct_correct') == 'True')
    judge_after_interactive_correct = sum(1 for r in rows if r.get('interactive_judge_after_debate_correct') == 'True')
    judge_after_non_interactive_correct = sum(1 for r in rows if r.get('non_interactive_judge_after_debate_correct') == 'True')
    
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    print(f"Total debates: {total}")
    print(f"Debater direct QA accuracy: {debater_correct}/{total} ({100*debater_correct/total:.1f}%)")
    print(f"Judge direct QA accuracy: {judge_direct_correct}/{total} ({100*judge_direct_correct/total:.1f}%)")
    print(f"Judge after interactive debate accuracy: {judge_after_interactive_correct}/{total} ({100*judge_after_interactive_correct/total:.1f}%)")
    print(f"Judge after non-interactive debate accuracy: {judge_after_non_interactive_correct}/{total} ({100*judge_after_non_interactive_correct/total:.1f}%)")
    print("="*70)


def tail_log_hint(log_files: List[Path]):
    """Print helpful commands for tailing log files."""
    print("\n" + "="*70)
    print("MONITORING DEBATES")
    print("="*70)
    print("To watch debates in real-time, open separate terminal windows and run:")
    print()
    for i, log_file in enumerate(log_files):
        print(f"  # Terminal {i+1}:")
        print(f"  tail -f {log_file}")
        print()
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple debates in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 2 debates and auto-open monitoring windows (macOS)
  python run_parallel_debates.py --num-debates 2 --auto-monitor
  # Creates: master_results_random_n2_20251016_120155.csv

  # Run 2 debates, manually monitor with tail -f
  python run_parallel_debates.py --num-debates 2
  # Then in separate terminals:
  # tail -f test_debate_results/debate_1_*.log
  # tail -f test_debate_results/debate_2_*.log

  # Run with specific seeds and wait for completion
  python run_parallel_debates.py --num-debates 3 --seeds 42 43 44 --wait
  # Creates: master_results_seed42_43_44_n3_20251016_120155.csv

  # Run with custom max-turns
  python run_parallel_debates.py --num-debates 2 --seeds 100 200 --max-turns 10
  # Creates: master_results_seed100_200_n2_turns10_20251016_120155.csv

  # Run quietly (less verbose output)
  python run_parallel_debates.py --num-debates 2 --quiet
        """
    )
    
    parser.add_argument('--num-debates', type=int, default=2,
                        help='Number of debates to run in parallel (default: 2)')
    parser.add_argument('--output-dir', type=str, default='./test_debate_results',
                        help='Output directory for results (default: ./test_debate_results)')
    parser.add_argument('--master-csv', type=str, default=None,
                        help='Path to master CSV for aggregated results (default: auto-generated descriptive name)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Random seeds for each debate (must match --num-debates)')
    parser.add_argument('--max-turns', type=int, default=20,
                        help='Maximum number of debate turns (default: 20)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output in debate processes')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for all debates to complete before exiting')
    parser.add_argument('--auto-monitor', action='store_true',
                        help='Automatically open terminal windows to monitor debates (macOS only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.seeds and len(args.seeds) != args.num_debates:
        parser.error(f"Number of seeds ({len(args.seeds)}) must match --num-debates ({args.num_debates})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate descriptive run folder name and master CSV filename
    if args.master_csv:
        master_csv = args.master_csv
        # Extract folder name from custom CSV path
        run_folder_name = Path(master_csv).stem
        run_output_dir = str(Path(args.output_dir) / run_folder_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build descriptive filename
        filename_parts = ['master_results']
        
        # Add seed info
        if args.seeds:
            seed_str = '_'.join(map(str, args.seeds))
            filename_parts.append(f'seed{seed_str}')
        else:
            filename_parts.append('random')
        
        # Add number of debates
        filename_parts.append(f'n{args.num_debates}')
        
        # Add max turns if not default
        if args.max_turns != 20:
            filename_parts.append(f'turns{args.max_turns}')
        
        # Add timestamp
        filename_parts.append(timestamp)
        
        run_folder_name = '_'.join(filename_parts)
        run_output_dir = str(Path(args.output_dir) / run_folder_name)
        master_csv = str(Path(run_output_dir) / f'{run_folder_name}.csv')
    
    # Create run-specific output directory
    os.makedirs(run_output_dir, exist_ok=True)
    
    print("="*70)
    print("PARALLEL DEBATE RUNNER")
    print("="*70)
    print(f"Number of debates: {args.num_debates}")
    print(f"Run folder: {run_output_dir}")
    print(f"Master CSV: {master_csv}")
    print(f"Max turns per debate: {args.max_turns}")
    print("="*70)
    
    # Note: No need to backup individual CSV since each run has its own folder
    
    # Launch debate processes
    processes = []
    log_files = []
    
    for i in range(args.num_debates):
        seed = args.seeds[i] if args.seeds else None
        process, log_file = run_single_debate_process(
            debate_id=i+1,
            output_dir=run_output_dir,
            seed=seed,
            max_turns=args.max_turns,
            quiet=args.quiet
        )
        processes.append((i+1, process))
        log_files.append(log_file)
    
    # Print monitoring hints
    tail_log_hint(log_files)
    
    # Auto-open monitoring windows if requested
    if args.auto_monitor:
        import platform
        if platform.system() == 'Darwin':  # macOS
            print("\nOpening monitoring windows...")
            for i, log_file in enumerate(log_files):
                cmd = f'tell application "Terminal" to do script "cd {os.getcwd()} && tail -f {log_file}"'
                subprocess.run(['osascript', '-e', cmd])
            print("✓ Monitoring windows opened!")
        else:
            print("\nWarning: --auto-monitor only supported on macOS")
            print("Please manually run the tail commands above")
    
    if not args.wait:
        print("\nDebates are running in the background.")
        print("Use the tail commands above to monitor progress.")
        print("\nTo wait for completion and see aggregated results, re-run with --wait flag")
        return
    
    # Wait for all processes to complete
    print("\nWaiting for all debates to complete...")
    print("(This may take several minutes)")
    print()
    
    completed = [False] * len(processes)
    
    while not all(completed):
        for idx, (debate_id, process) in enumerate(processes):
            if not completed[idx]:
                retcode = process.poll()
                if retcode is not None:
                    completed[idx] = True
                    if retcode == 0:
                        print(f"[Debate {debate_id}] ✓ Completed successfully")
                    else:
                        print(f"[Debate {debate_id}] ✗ Failed with exit code {retcode}")
        
        if not all(completed):
            time.sleep(1)
    
    print("\n" + "="*70)
    print("All debates completed!")
    print("="*70)
    
    # Aggregate results
    print("\nAggregating results...")
    aggregate_results(run_output_dir, master_csv)
    
    # Print statistics
    print_aggregate_stats(master_csv)
    
    # List generated files
    print("\nGenerated files:")
    print(f"  Master CSV: {master_csv}")
    for log_file in log_files:
        print(f"  Log: {log_file}")
    
    # List detail files
    detail_files = sorted(Path(run_output_dir).glob('debate_detail_*.txt'))
    if detail_files:
        print(f"\n  Detail files ({len(detail_files)}):")
        for detail_file in detail_files[-args.num_debates:]:  # Show only recent ones
            print(f"    {detail_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

