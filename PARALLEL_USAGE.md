# Running Parallel Debates

## Quick Start

### Run 2 debates simultaneously (wait for completion):
```bash
python run_parallel_debates.py --num-debates 2 --wait
```

### Run debates in background (non-blocking):
```bash
python run_parallel_debates.py --num-debates 2
```

### Note about monitoring:
```bash
# Debates run in the background
# Results are saved to debate_detail_*.txt files as they complete
# Use --wait flag to see aggregate statistics when all debates finish
```

## Features

✅ **Run multiple debates simultaneously** - Leverage parallel processing  
✅ **Master CSV aggregation** - All results combined in one file  
✅ **Individual detail files** - Each debate saves its own detailed output  
✅ **Reproducible** - Use `--seeds` for consistent results  
✅ **Efficient** - No redundant log files, all info in detail files  

## Command Options

```bash
python run_parallel_debates.py [OPTIONS]

Options:
  --num-debates N       Number of parallel debates (default: 2)
  --max-concurrent N    Max concurrent debates (default: all at once)
                        Use this to avoid API rate limits
  --output-dir DIR      Output directory (default: ./test_debate_results)
  --master-csv FILE     Master CSV path (default: OUTPUT_DIR/master_results_summary.csv)
  --seed N              Random seed for reproducibility (generates deterministic
                        seeds for each debate)
  --max-turns N         Max debate turns per debate (default: 20)
  --quiet               Suppress verbose output in logs
  --wait                Wait for completion before exiting
```

## Examples

### Run 100 debates with rate limiting (recommended):
```bash
python run_parallel_debates.py --num-debates 100 --max-concurrent 20 --wait
```
This runs 100 debates in batches of 20, preventing API rate limit errors.

### Run 100 debates reproducibly with a seed:
```bash
python run_parallel_debates.py --num-debates 100 --seed 42 --max-concurrent 20 --wait
```
The seed 42 generates deterministic seeds for all 100 debates, making results reproducible.

### Run quietly (less verbose logs):
```bash
python run_parallel_debates.py --num-debates 2 --quiet --wait
```

### Custom output directory:
```bash
python run_parallel_debates.py --num-debates 2 --output-dir ./my_experiments --wait
```

## Output Files

After running, you'll find in the run-specific folder:

1. **Results CSV**: `test_debate_results/master_results_RUNINFO/master_results_RUNINFO.csv`
   - All results from all debates in this run
   - Example: `master_results_random_n100_20251016_132254.csv`
   - Contains debater/judge direct QA accuracy, debate outcomes, confidence scores

2. **Detail files**: `test_debate_results/master_results_RUNINFO/debate_detail_RUNID_TIMESTAMP_QIDX.txt`
   - Complete debate transcript and analysis
   - One per debate
   - Contains all the information (questions, arguments, verdicts, reasoning)

## Workflow Example

```bash
# Start debates and wait for completion
python run_parallel_debates.py --num-debates 2 --wait

# Or run in background (non-blocking)
python run_parallel_debates.py --num-debates 2

# Check results in the generated folder
ls test_debate_results/master_results_random_n2_*/
# You'll see: CSV files and debate_detail_*.txt files

# Check results
cat test_debate_results/master_results_random_n2_*/*.csv

# Read full debate transcripts
cat test_debate_results/master_results_random_n2_*/debate_detail_*.txt
```

## Original Single Debate Script

The original `run_single_debate.py` still works independently:

```bash
python run_single_debate.py --seed 42 --output-dir ./my_debate
```

Both scripts are maintained and fully compatible!

