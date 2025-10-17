# Migration to JSONL Format - Summary

**Date:** October 17, 2025  
**Status:** ✅ Complete

## What Changed

Replaced CSV format with JSONL (JSON Lines) for storing debate results.

## Files Modified

### Core Changes
1. **run_single_debate.py**
   - Renamed `save_results_to_csv()` → `save_results_to_jsonl()`
   - Changed output format to nested JSON structure
   - Updated argument `--csv-filename` → `--jsonl-filename`
   - Default filename: `master_results.jsonl`

2. **run_parallel_debates.py**
   - Updated `print_aggregate_stats()` to read JSONL
   - Changed all `master_csv` variables → `master_jsonl`
   - Updated argument `--master-csv` → `--master-jsonl`
   - Changed import from `csv` to `json`

### New Files Created
1. **analyze_jsonl.py** - Analysis helper functions
   - `load_results()` - Load JSONL file
   - `results_to_dataframe()` - Convert to pandas
   - `filter_by_mode()` - Filter by debate mode
   - `compare_modes()` - Compare modes within a run
   - `compare_runs()` - Compare different runs
   - `print_summary()` - Quick stats

2. **convert_csv_to_jsonl.py** - Migration tool
   - Convert single CSV file to JSONL
   - Batch convert all CSVs in directory
   - Preserves all data from old format

3. **JSONL_FORMAT.md** - Documentation
   - Format specification
   - Usage examples
   - Analysis workflows

## New JSONL Structure

### Old CSV Format (Fixed Columns)
```csv
run_id,timestamp,question_idx,debater_direct_correct,debater_confidence,debater_cached,judge_direct_correct,judge_confidence,judge_cached,interactive_debate_turns,interactive_judge_verdict_winner,interactive_judge_after_debate_correct,interactive_judge_after_debate_confidence,non_interactive_debate_turns,non_interactive_judge_verdict_winner,non_interactive_judge_after_debate_correct,non_interactive_judge_after_debate_confidence
3a8c9032,20251017_114044,176,False,90,True,False,90,True,,,,,3,A,False,85
```
❌ Empty columns when mode not run  
❌ Hard to add new modes  
❌ Flat structure  

### New JSONL Format (Flexible Schema)
```json
{
  "run_id": "3a8c9032",
  "timestamp": "20251017_114044",
  "question_idx": 176,
  "debater_direct": {"correct": false, "confidence": 90, "cached": true},
  "judge_direct": {"correct": false, "confidence": 90, "cached": true},
  "modes": {
    "non_interactive": {"turns": 3, "winner": "A", "correct": false, "confidence": 85}
  }
}
```
✅ Only save modes that were run  
✅ Easy to add new modes  
✅ Clean nested structure  
✅ Future-proof  

## Benefits

1. **Flexible Schema**
   - Different runs can have different modes
   - No empty columns for unused modes
   - Easy to add new debate variants

2. **Better Analysis**
   - Load into pandas with one function
   - Filter by mode naturally
   - Compare runs easily
   - Standard JSON tools (jq, etc.)

3. **More Maintainable**
   - Cleaner code structure
   - Self-documenting format
   - Easy to extend

4. **No Breaking Changes Needed**
   - Old CSV files can be converted
   - Conversion script provided
   - All functionality preserved

## Testing

✅ Single debate run completed successfully  
✅ JSONL file generated correctly  
✅ Aggregate stats working  
✅ Analysis tools tested  
✅ CSV to JSONL conversion tested  

## Migration Path

For existing CSV files:
```bash
# Convert single file
python convert_csv_to_jsonl.py path/to/master_results.csv

# Convert entire directory
python convert_csv_to_jsonl.py parallel_debate_runs/
```

## Example Usage

### Running debates (same as before)
```bash
python run_parallel_debates.py --num-debates 100 --seed 42 --wait
```

### Analyzing results (new!)
```bash
# Quick summary
python analyze_jsonl.py parallel_debate_runs/master_results_seed42_n100_20251017/master_results.jsonl

# Python analysis
from analyze_jsonl import load_results, results_to_dataframe
results = load_results('master_results.jsonl')
df = results_to_dataframe(results)
```

## Future Extensions

Now easy to add:
- New debate modes (e.g., "multi_round", "blind_judge", "hypothetical")
- Additional metadata (model temperature, token counts, etc.)
- Experiment variants without changing schema
- A/B tests with different configurations

## Backward Compatibility

⚠️ **No backward compatibility** - clean break from CSV format  
✅ Migration tool provided to convert existing data  
✅ All existing analysis can be replicated with new tools  

