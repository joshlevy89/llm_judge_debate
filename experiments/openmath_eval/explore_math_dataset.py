#!/usr/bin/env python3

import argparse
from datasets import load_dataset


def print_math_samples(subset: str, levels: list[int], n_samples: int = 5):
    """
    Print n samples from the MATH dataset for a given subset and levels.
    
    Args:
        subset: Subject area (e.g., 'algebra', 'counting_and_probability', 'geometry', 
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus')
        levels: List of difficulty levels (1-5)
        n_samples: Number of samples to print per level
    """
    dataset = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
    
    for level in levels:
        filtered = [item for item in dataset if item['level'] == f"Level {level}"]
        
        print(f"\n{'='*80}")
        print(f"Subset: {subset} | Level: {level} | Total matching: {len(filtered)}")
        print(f"{'='*80}\n")
        
        for i, item in enumerate(filtered[:n_samples], 1):
            print(f"Sample {i}/{min(n_samples, len(filtered))}")
            print(f"{'-'*80}")
            print(f"Problem:\n{item['problem']}\n")
            # print(f"Solution:\n{item['solution']}\n")
            # print(f"Answer: {item['answer']}")
            print(f"{'-'*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore MATH dataset samples")
    parser.add_argument("--subset", "-s", type=str, required=True,
                        help="Subset name (e.g., algebra, geometry)")
    parser.add_argument("--levels", "-l", type=str, required=True,
                        help="Comma-separated difficulty levels (e.g., '1,3' or '1,2,3')")
    parser.add_argument("--n_samples", "-n", type=int, default=5,
                        help="Number of samples to print per level (default: 5)")
    
    args = parser.parse_args()
    
    levels = [int(level.strip()) for level in args.levels.split(',')]
    for level in levels:
        if level not in [1, 2, 3, 4, 5]:
            parser.error(f"Invalid level: {level}. Must be between 1 and 5.")
    
    print_math_samples(args.subset, levels, args.n_samples)
