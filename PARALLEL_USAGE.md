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

### Monitor live output in separate terminals:
```bash
# After starting debates, use the tail commands printed by the script:
tail -f test_debate_results/debate_1_*.log
tail -f test_debate_results/debate_2_*.log
```

### Auto-open monitoring windows (macOS):
```bash
# First, start the debates
python run_parallel_debates.py --num-debates 2

# Then open monitoring windows
./monitor_debates.sh test_debate_results/debate_1_*.log test_debate_results/debate_2_*.log
```

## Features

✅ **Run multiple debates simultaneously** - Leverage parallel processing  
✅ **Separate log files** - Each debate has its own complete log  
✅ **Master CSV aggregation** - All results combined in one file  
✅ **Individual detail files** - Each debate saves its own detailed output  
✅ **Real-time monitoring** - Tail log files to watch progress  
✅ **Reproducible** - Use `--seeds` for consistent results  

## Command Options

```bash
python run_parallel_debates.py [OPTIONS]

Options:
  --num-debates N       Number of parallel debates (default: 2)
  --output-dir DIR      Output directory (default: ./test_debate_results)
  --master-csv FILE     Master CSV path (default: OUTPUT_DIR/master_results_summary.csv)
  --seeds S1 S2 ...     Random seeds for each debate (must match --num-debates)
  --max-turns N         Max debate turns per debate (default: 20)
  --quiet               Suppress verbose output in logs
  --wait                Wait for completion before exiting
```

## Examples

### Run 3 debates with specific seeds:
```bash
python run_parallel_debates.py --num-debates 3 --seeds 42 43 44 --wait
```

### Run quietly (less verbose logs):
```bash
python run_parallel_debates.py --num-debates 2 --quiet --wait
```

### Custom output directory:
```bash
python run_parallel_debates.py --num-debates 2 --output-dir ./my_experiments --wait
```

## Output Files

After running, you'll find:

1. **Master CSV**: `test_debate_results/master_results_summary.csv`
   - Aggregated results from all debates
   - Cumulative across all runs

2. **Individual CSVs**: `test_debate_results/debate_results_summary.csv`
   - Results from current run only
   - Gets backed up before each new run

3. **Log files**: `test_debate_results/debate_N_TIMESTAMP.log`
   - Full console output for each debate
   - Useful for debugging and real-time monitoring

4. **Detail files**: `test_debate_results/debate_detail_TIMESTAMP_QIDX.txt`
   - Complete debate transcript and analysis
   - One per debate

## Workflow Example

```bash
# Terminal 1: Start 2 debates
python run_parallel_debates.py --num-debates 2

# Terminal 2: Monitor debate 1
tail -f test_debate_results/debate_1_*.log

# Terminal 3: Monitor debate 2
tail -f test_debate_results/debate_2_*.log

# Later: Check aggregated results
cat test_debate_results/master_results_summary.csv
```

## Original Single Debate Script

The original `run_single_debate.py` still works independently:

```bash
python run_single_debate.py --seed 42 --output-dir ./my_debate
```

Both scripts are maintained and fully compatible!

