#!/usr/bin/env python3

import argparse
import random
from datasets import load_dataset

ALL_SUBSETS = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
]
ALL_LEVELS = [1, 2, 3, 4, 5]


def print_math_samples(subset: str, levels: list[int], n_samples: int = 5):
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
            print(f"{'-'*80}\n")


def print_random_samples(subsets: list[str], levels: list[int], n_samples: int):
    all_samples = []
    for subset in subsets:
        dataset = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
        for item in dataset:
            try:
                level_num = int(item['level'].replace("Level ", ""))
            except ValueError:
                continue
            if level_num in levels:
                all_samples.append({**item, 'subset': subset})
    
    print(f"Total matching samples: {len(all_samples)}")
    samples = random.sample(all_samples, min(n_samples, len(all_samples)))
    
    for i, item in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"Sample {i}/{len(samples)} | Subset: {item['subset']} | {item['level']}")
        print(f"{'='*80}")
        print(f"Problem:\n{item['problem']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore MATH dataset samples")
    parser.add_argument("--subset", "-s", type=str, default=None,
                        help="Subset name (e.g., algebra, geometry). If not specified, uses all subsets.")
    parser.add_argument("--levels", "-l", type=str, default=None,
                        help="Comma-separated difficulty levels (e.g., '1,3'). If not specified, uses all levels 1-5.")
    parser.add_argument("--n_samples", "-n", type=int, default=5,
                        help="Number of samples to print (default: 5)")
    parser.add_argument("--total", "-t", action="store_true",
                        help="Sample n total at random instead of n per subset/level")
    
    args = parser.parse_args()
    
    subsets = ALL_SUBSETS if args.subset is None else [args.subset]
    
    if args.levels is None:
        levels = ALL_LEVELS
    else:
        levels = [int(level.strip()) for level in args.levels.split(',')]
        for level in levels:
            if level not in ALL_LEVELS:
                parser.error(f"Invalid level: {level}. Must be between 1 and 5.")
    
    if args.total:
        print_random_samples(subsets, levels, args.n_samples)
    else:
        for subset in subsets:
            print_math_samples(subset, levels, args.n_samples)
