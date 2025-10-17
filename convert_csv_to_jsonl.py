#!/usr/bin/env python3
"""
Convert existing CSV results to JSONL format.

This script helps migrate old debate results from CSV to the new JSONL format.
"""

import csv
import json
from pathlib import Path
import argparse


def convert_csv_to_jsonl(csv_path: str, jsonl_path: str = None):
    """
    Convert a CSV results file to JSONL format.
    
    Args:
        csv_path: Path to input CSV file
        jsonl_path: Path to output JSONL file (default: same name with .jsonl extension)
    """
    csv_path = Path(csv_path)
    
    if jsonl_path is None:
        jsonl_path = csv_path.with_suffix('.jsonl')
    else:
        jsonl_path = Path(jsonl_path)
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Converting {len(rows)} results from CSV to JSONL...")
    
    # Convert each row to new JSONL format
    results = []
    for row in rows:
        result = {
            'run_id': row['run_id'],
            'timestamp': row['timestamp'],
            'question_idx': int(row['question_idx']),
            'debater_direct': {
                'correct': row['debater_direct_correct'].lower() == 'true',
                'confidence': int(row['debater_confidence']) if row['debater_confidence'] else None,
                'cached': row['debater_cached'].lower() == 'true' if row.get('debater_cached') else False,
            },
            'judge_direct': {
                'correct': row['judge_direct_correct'].lower() == 'true',
                'confidence': int(row['judge_confidence']) if row['judge_confidence'] else None,
                'cached': row['judge_cached'].lower() == 'true' if row.get('judge_cached') else False,
            },
            'modes': {}
        }
        
        # Add interactive mode if present
        if row.get('interactive_debate_turns') and row['interactive_debate_turns']:
            result['modes']['interactive'] = {
                'turns': int(row['interactive_debate_turns']),
                'winner': row['interactive_judge_verdict_winner'],
                'correct': row['interactive_judge_after_debate_correct'].lower() == 'true' if row['interactive_judge_after_debate_correct'] else None,
                'confidence': int(row['interactive_judge_after_debate_confidence']) if row['interactive_judge_after_debate_confidence'] else None,
            }
        
        # Add non-interactive mode if present
        if row.get('non_interactive_debate_turns') and row['non_interactive_debate_turns']:
            result['modes']['non_interactive'] = {
                'turns': int(row['non_interactive_debate_turns']),
                'winner': row['non_interactive_judge_verdict_winner'],
                'correct': row['non_interactive_judge_after_debate_correct'].lower() == 'true' if row['non_interactive_judge_after_debate_correct'] else None,
                'confidence': int(row['non_interactive_judge_after_debate_confidence']) if row['non_interactive_judge_after_debate_confidence'] else None,
            }
        
        results.append(result)
    
    # Write JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Converted {len(results)} results")
    print(f"Output: {jsonl_path}")


def convert_directory(directory: str):
    """
    Convert all CSV files in a directory to JSONL.
    
    Args:
        directory: Path to directory containing CSV files
    """
    directory = Path(directory)
    csv_files = list(directory.rglob('*.csv'))
    
    print(f"Found {len(csv_files)} CSV files in {directory}")
    
    for csv_file in csv_files:
        print(f"\n{csv_file.name}:")
        convert_csv_to_jsonl(str(csv_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CSV results to JSONL format')
    parser.add_argument('path', type=str,
                        help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL path (only for single file conversion)')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        convert_csv_to_jsonl(str(path), args.output)
    elif path.is_dir():
        if args.output:
            print("Warning: --output is ignored when converting a directory")
        convert_directory(str(path))
    else:
        print(f"Error: {path} is not a file or directory")
        exit(1)

