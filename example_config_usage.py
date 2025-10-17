#!/usr/bin/env python3
"""
Example script showing how to use config_used.json for reproducibility and analysis.
"""

import json
from pathlib import Path

# Example 1: Load and inspect a config
run_dir = Path("parallel_debate_runs/run_20251017_121343")
config_path = run_dir / "config_used.json"

if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Configuration from run:")
    print(f"  Debate Model: {config['model_configuration']['debate_model']}")
    print(f"  Judge Model: {config['model_configuration']['judge_model']}")
    print(f"  Max Turns: {config['debate_configuration']['max_turns']}")
    print(f"  Debate Mode: {config['debate_configuration']['debate_mode']}")
    print(f"  Temperature: {config['temperature_settings']['direct_qa_temperature']}")
    print(f"  Timestamp: {config['timestamp']}")
    print()

# Example 2: Compare configs across multiple runs
def compare_runs(run_dirs):
    """Compare configurations across multiple runs."""
    configs = []
    for run_dir in run_dirs:
        config_path = Path(run_dir) / "config_used.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                configs.append({
                    'run': run_dir.name if isinstance(run_dir, Path) else Path(run_dir).name,
                    'config': config
                })
    
    # Compare key settings
    print("Comparison across runs:")
    print("-" * 60)
    for item in configs:
        print(f"\nRun: {item['run']}")
        print(f"  Models: {item['config']['model_configuration']['debate_model']} / "
              f"{item['config']['model_configuration']['judge_model']}")
        print(f"  Max Turns: {item['config']['debate_configuration']['max_turns']}")
        print(f"  Mode: {item['config']['debate_configuration']['debate_mode']}")

# Example 3: Reproduce a run with same config
def reproduce_run_command(config_path):
    """Generate command to reproduce a run."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    cmd_parts = ["python run_parallel_debates.py"]
    
    # Add relevant arguments
    runtime = config['runtime_arguments']
    if runtime.get('seed') is not None:
        cmd_parts.append(f"--seed {runtime['seed']}")
    
    debate_config = config['debate_configuration']
    if debate_config.get('max_turns_overridden'):
        cmd_parts.append(f"--max-turns {debate_config['max_turns']}")
    
    if runtime.get('quiet'):
        cmd_parts.append("--quiet")
    
    print("\nCommand to reproduce this run:")
    print(" ".join(cmd_parts))

if config_path.exists():
    reproduce_run_command(config_path)

