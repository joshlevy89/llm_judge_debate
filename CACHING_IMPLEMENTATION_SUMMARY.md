# Baseline Caching Implementation Summary

## What Was Implemented

A smart caching system that stores direct QA baseline results to avoid re-running them when experimenting with different debate parameters.

## Files Modified/Created

### New Files
1. **`baseline_cache.py`** - Core caching module with functions:
   - `get_cached_qa()` - Retrieve cached results
   - `save_qa_to_cache()` - Store new results
   - `clear_cache()` - Clear cache files
   - `get_cache_stats()` - View cache statistics

2. **`BASELINE_CACHE.md`** - Complete documentation of the caching system

3. **`CACHING_IMPLEMENTATION_SUMMARY.md`** - This file

### Modified Files
1. **`config.py`** - Added configuration options:
   ```python
   USE_BASELINE_CACHE = True
   SAVE_TO_BASELINE_CACHE = True
   BASELINE_CACHE_DIR = './baseline_cache'
   ```

2. **`run_single_debate.py`** - Integrated caching:
   - Import baseline_cache module
   - Check cache before running direct QA
   - Save results to cache after running
   - Add `cached` flag to results
   - Display `[CACHED]` indicator in output

3. **CSV Output** - Added two new columns:
   - `debater_cached` (bool) - Whether debater baseline was cached
   - `judge_cached` (bool) - Whether judge baseline was cached

## How It Works

### Cache Structure
- **Location**: `./baseline_cache/`
- **Format**: JSON files, one per model/type
- **Naming**: `baseline_{type}_{model}_temp{temp}.json`
- **Keys**: Model name, question index, temperature, **option values (A & B)**

### First Run (Cache Miss)
1. Check cache for question - not found
2. Run direct QA via API
3. Save result to cache
4. Mark as `cached: false` in CSV

### Subsequent Runs (Cache Hit)
1. Check cache for question - found!
2. Use cached result (no API call)
3. Mark as `cached: true` in CSV
4. Show `[Using cached result]` in output

### Cache Invalidation
Automatic when:
- Model changes (separate cache per model)
- Temperature changes (included in cache key)
- **Choice order changes** (option A/B values don't match)
- Manual deletion of cache files

## Benefits

1. **Zero Wasted API Calls** - Never re-run stable baselines when experimenting
2. **Always Available** - Automatically runs and caches if missing
3. **Transparent** - CSV tracks what was cached vs fresh
4. **Model-Aware** - Each model has separate cache
5. **Temperature-Aware** - Cache respects temperature settings

## Testing Results

Tested with two consecutive runs:
- **Run 1**: Fresh baselines (`debater_cached: False`, `judge_cached: False`)
- **Run 2**: Cached baselines (`debater_cached: True`, `judge_cached: True`)

Both runs produced identical baseline results, but Run 2 skipped the API calls.

## Example Use Cases

### Experiment with Debate Parameters
```bash
# First run establishes baselines
python run_parallel_debates.py --num-debates 100 --seed 42

# Experiment with different parameters
# Baselines reused, only debates run fresh
python run_parallel_debates.py --num-debates 100 --seed 42 --max-turns 10
python run_parallel_debates.py --num-debates 100 --seed 42 --max-turns 5
```

### Compare Debate Modes
```bash
# All use same cached baselines
python run_parallel_debates.py --num-debates 50 --seed 42  # Both modes
# Future: Could add --debate-mode flag to run only one mode
```

## Configuration Options

Enable/disable as needed in `config.py`:

```python
# Use cached results (recommended: True)
USE_BASELINE_CACHE = True

# Save new results to cache (recommended: True)  
SAVE_TO_BASELINE_CACHE = True

# Change cache location if needed
BASELINE_CACHE_DIR = './baseline_cache'
```

## Cache Management

### View Cache
```bash
ls -lh baseline_cache/
cat baseline_cache/baseline_debater_gpt-4o-mini_temp0.0.json
```

### Clear Cache
```python
import baseline_cache
baseline_cache.clear_cache()  # Clear all
baseline_cache.clear_cache('debater')  # Clear debater only
```

Or manually:
```bash
rm -rf baseline_cache/
```

## Performance Impact

- **Cache Hit**: ~0ms (instant, no API call)
- **Cache Miss**: Normal API latency + ~50ms for cache write
- **Storage**: ~500 bytes per cached question
- **Typical Savings**: 2 API calls per question when cache is warm

## Future Enhancements (Optional)

1. **Config flag for debate modes** - Run only interactive or non-interactive
2. **CLI flag to disable cache** - `--no-cache` for one-off runs
3. **Cache stats command** - View cache hit rates and coverage
4. **Shared cache** - Sync cache across team/machines (via git)
5. **Dataset versioning** - Track which dataset version was used

## Notes

- Cache assumes questions are stable (same index = same question)
- Cache is local to machine (not synced automatically)
- Cache files are human-readable JSON
- Safe to delete cache anytime - will rebuild automatically
- Consider committing cache to git for reproducibility
- **Choice order matters**: Cache validates that option A and B match exactly. This allows you to experiment with different choice orderings - each ordering is treated as a separate experiment and will trigger fresh QA runs when the order changes.

## Migration

No migration needed! The system:
- Works immediately on next run
- Detects missing cache and creates it
- Backward compatible (old CSVs work fine)
- Optional (can be disabled in config)

