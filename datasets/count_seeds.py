"""Quick script to count available seeds in MATH dataset."""
from datasets import load_dataset

LEVELS = [1, 2, 3, 4, 5]
SUBJECTS = ["number_theory"]

print("Counting available seeds in MATH dataset...")
for subject in SUBJECTS:
    print(f"\nSubject: {subject}")
    dataset = load_dataset("EleutherAI/hendrycks_math", subject)
    
    level_counts = {level: 0 for level in LEVELS}
    
    for split in ["train", "test"]:
        print(f"  {split} split:")
        for item in dataset[split]:
            level_str = item.get("level", "")
            try:
                level = int(level_str.replace("Level ", "")) if "Level " in level_str else None
            except ValueError:
                continue
            if level in LEVELS:
                level_counts[level] += 1
        print(f"    Found {sum(level_counts.values())} problems so far")
    
    print(f"\n  Total by level:")
    for level in LEVELS:
        print(f"    Level {level}: {level_counts[level]} seeds")
