# File Naming and Matching Strategy

## Overview

All debate runs now use a **unique run identifier (run_id)** that appears in:
1. Detail filenames  
2. CSV as a column
3. Detail file headers

This makes it easy to match files across all outputs.

## Naming Format

### Detail Files
```
debate_detail_{run_id}_{timestamp}_{question_idx}.txt
```

**Examples:**
- `debate_detail_a3f5b7c2_20251016_123050_42.txt` (run_id: a3f5b7c2, question_idx: 42)
- `debate_detail_d8e9a1b4_20251016_123051_163.txt` (run_id: d8e9a1b4, question_idx: 163)

### Results CSV
```
master_results_{seed_info}_n{num_debates}_{turns}_{timestamp}/
  master_results_{seed_info}_n{num_debates}_{turns}_{timestamp}.csv
```

**Examples:**
- `master_results_seed100_200_n2_20251016_123045/master_results_seed100_200_n2_20251016_123045.csv`
- `master_results_random_n3_turns10_20251016_123045/master_results_random_n3_turns10_20251016_123045.csv`

Each run creates its own folder with a descriptive name containing the results CSV and all detail files.

## CSV Column: run_id

The `run_id` appears as the **first column** in all CSVs:

```csv
run_id,timestamp,question_idx,debater_direct_correct,...
a3f5b7c2,20251016_123050,42,True,...
d8e9a1b4,20251016_123051,163,False,...
```

## Matching Files

### To find all files for a specific debate run:

```bash
# Get run_id from CSV
run_id="a3f5b7c2"

# Find corresponding files
ls test_debate_results/*${run_id}*

# Result:
# debate_detail_a3f5b7c2_20251016_123045_42.txt
# debate_detail_a3f5b7c2_20251016_123050_42.txt
```

### Using grep to find a CSV row's files:

```bash
# Find a specific result in CSV
grep "a3f5b7c2" test_debate_results/master_results*.csv

# Then find all related files
ls test_debate_results/*a3f5b7c2*
```

### Python matching example:

```python
import pandas as pd
import glob

# Read master CSV
df = pd.read_csv('test_debate_results/master_results_*.csv')

# Get run_id for a specific row
run_id = df.iloc[0]['run_id']

# Find corresponding detail file
detail_files = glob.glob(f'test_debate_results/*/debate_detail_{run_id}_*.txt')

print(f"Run ID: {run_id}")
print(f"Detail: {detail_files[0]}")
```

## Run ID Format

- **8-character UUID prefix** (e.g., `a3f5b7c2`)
- Unique across all runs
- Short enough for readability
- Unlikely to collide (16^8 = 4.3 billion combinations)

## Benefits

✅ **Reliable Matching**: No ambiguity about which files belong together  
✅ **Parallel-Safe**: Multiple runs can happen simultaneously without confusion  
✅ **Human-Readable**: Short IDs are easy to work with  
✅ **Persistent**: IDs remain stable across file moves or renames  
✅ **Traceable**: Can trace from CSV → log → detail file and vice versa  

## Console Output

When running parallel debates, you'll see:

```
[Debate 1] Starting... (run_id: a3f5b7c2)
[Debate 2] Starting... (run_id: d8e9a1b4)
```

And in the save output:

```
7. Saving results...
   Run ID: a3f5b7c2
   CSV: test_debate_results/debate_results_summary.csv
   Details: test_debate_results/debate_detail_a3f5b7c2_20251016_123050_42.txt
```

## Detail File Header

Each detail file shows its run_id prominently:

```
================================================================================
DEBATE ANALYSIS RESULTS
Run ID: a3f5b7c2
Generated: 2025-10-16 12:30:50
================================================================================
```

This makes it easy to verify you're looking at the right file!

