# Baseline Cache System

## Overview

The baseline cache system stores direct QA results (before debates) to avoid re-running stable baselines when experimenting with different debate parameters. This saves API calls and time while ensuring baseline comparisons are always available.

## How It Works

### Cache Storage

- **Location**: `./baseline_cache/` directory
- **Format**: JSON files, one per model/type combination
- **Naming**: `baseline_{model_type}_{model_name}_temp{temperature}.json`
  - Example: `baseline_debater_gemini-2.5-flash_temp0.0.json`
  - Example: `baseline_judge_gpt-3.5-turbo_temp0.0.json`

### Cache Keys

Results are keyed and validated by:
1. **Model name** - Different models have separate cache files
2. **Model type** - 'debater' or 'judge'
3. **Question index** - Each question has its own cache entry (based on dataset index)
4. **Temperature** - Ensures consistency with current settings
5. **Dataset info** - Dataset name, subset, and split (validates cache is for correct dataset)
6. **Option values** - The exact text of option A and option B (validates choice order hasn't changed)

### Cache Behavior

**On first run** of a question with specific option order:
1. Check cache for existing result with matching option order
2. If not found, run direct QA via API
3. Save result to cache (including option A and B values)
4. Mark as `cached: false` in JSONL

**On subsequent runs** of the same question with same option order:
1. Check cache for existing result
2. If found and all parameters match (temperature, options), use cached result
3. Skip API call
4. Mark as `cached: true` in JSONL

**If option order changes:**
1. Cache validation fails (options don't match cached values)
2. Warning message: "Cached result has different options"
3. Run fresh QA with new option order
4. Save as new cache entry (overwrites previous for this question_idx)

### Configuration

In `config.py`:

```python
# Enable/disable cache lookups
USE_BASELINE_CACHE = True  # Check cache before running QA

# Enable/disable cache writes
SAVE_TO_BASELINE_CACHE = True  # Save new QA results to cache

# Cache directory location
BASELINE_CACHE_DIR = './baseline_cache'
```

## Benefits

### 1. **No Wasted API Calls**
When experimenting with debate parameters (interactive vs non-interactive, max_turns, etc.), you don't re-run direct QA baselines that won't change.

### 2. **Always Available Baselines**
If cache is missing for a question, it automatically runs and caches it. You never lose baseline comparisons.

### 3. **Model-Aware**
Changing models automatically invalidates cache. Each model has separate cache files.

### 4. **Temperature-Aware**
If you change `DIRECT_QA_TEMPERATURE`, cached results are ignored and fresh results are generated.

### 5. **Transparent**
CSV columns (`debater_cached`, `judge_cached`) show whether results were cached or fresh.

## CSV Columns

New columns track cache usage:

```csv
debater_cached,judge_cached
False,False          # Fresh results, saved to cache
True,True            # Both from cache
False,True           # Debater fresh, judge cached
```

## Example Use Cases

### Experiment with Debate Modes
```bash
# First run - establishes baselines
python run_parallel_debates.py --num-debates 10 --seed 42

# Experiment with different debate modes
# Baselines are reused from cache, only debates run
python run_parallel_debates.py --num-debates 10 --seed 42 --max-turns 10
python run_parallel_debates.py --num-debates 10 --seed 42 --max-turns 5
```

### Change Models
```bash
# Run with gemini-2.5-flash as debater
python run_parallel_debates.py --num-debates 10 --seed 42

# Change to different model in config.py
# Cache is model-specific, so fresh baselines run for new model
python run_parallel_debates.py --num-debates 10 --seed 42
```

## Cache Management

### View Cache Stats

The cache files are human-readable JSON. Example structure:

```json
{
  "metadata": {
    "model_name": "gemini-2.5-flash",
    "model_type": "debater",
    "temperature": 0.0,
    "dataset_name": "Idavidrein/gpqa",
    "dataset_subset": "gpqa_diamond",
    "dataset_split": "train",
    "created": "2025-10-17T10:00:00",
    "last_updated": "2025-10-17T11:30:00"
  },
  "results": {
    "42": {
      "selected_letter": "A",
      "selected_answer": "Water",
      "is_correct": true,
      "confidence": 85,
      "reasoning": "...",
      "temperature": 0.0,
      "option_a": "Water",
      "option_b": "Hydrogen peroxide",
      "timestamp": "2025-10-17T10:05:00"
    }
  }
}
```

### Clear Cache

To force fresh baselines:

```python
# In Python
import baseline_cache

# Clear all cache
baseline_cache.clear_cache()

# Clear only debater cache
baseline_cache.clear_cache('debater')

# Clear only judge cache
baseline_cache.clear_cache('judge')
```

Or manually delete files:
```bash
rm -rf baseline_cache/
```

### Disable Cache

To temporarily disable caching without deleting cache files:

```python
# In config.py
USE_BASELINE_CACHE = False  # Don't use cache, always run fresh
SAVE_TO_BASELINE_CACHE = False  # Don't save to cache
```

## Notes

### Cache Consistency

The cache assumes:
- Questions at the same index are the same across runs (within the same dataset)
- Direct QA with temperature 0.0 is deterministic enough to cache
- Model behavior is consistent for the same question and option order

**Question Index:** The cache keys are based on the actual index in the GPQA diamond dataset (`dataset[42]` always refers to the same question).

**Choice Order Sensitivity:** The cache validates that option A and option B are the same as when the cached result was generated. This means:
- If you run the same question with different randomization seeds, you'll get cache hits only when the choice order matches
- This allows you to experiment with choice order effects - each unique ordering is treated as a separate cache entry
- The cache entry for a question_idx is overwritten when a new option ordering is used (only the most recent option configuration is cached per question)

### Cache Invalidation

Cache is automatically invalidated (ignored) when:
- **Model name changes** - Different models have separate cache files
- **Temperature changes** - Cached results won't match current temperature
- **Dataset changes** - If you change `DATASET_NAME`, `DATASET_SUBSET`, or `DATASET_SPLIT` in `config.py`, cache will be ignored
- **Choice order changes** - If option A and option B values don't match the cached values, cache is ignored

When cache is invalidated, you'll see a warning message and fresh results will be generated.

### Performance

- **Cache hit**: ~0ms (instant)
- **Cache miss**: Normal API latency + cache write (~100ms)
- **Storage**: ~500 bytes per cached question

### Limitations

- Cache stores question index, not full question text (more compact, but relies on dataset stability)
- If you manually modify the dataset source (rare), you should clear the cache
- Cache doesn't sync across machines (local only)
- Cache validation checks dataset name/subset/split but not dataset version (GPQA is stable, so this is typically fine)

## Best Practices

1. **Leave caching enabled** (default) for normal use
2. **Clear cache** when changing datasets or major model updates
3. **Review `*_cached` columns** in CSV to understand what ran fresh vs cached
4. **Backup cache files** if you want to preserve baselines across machines
5. **Commit cache files to git** if you want team-wide baseline consistency (optional)

