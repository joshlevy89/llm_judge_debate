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

Results are keyed by:
1. **Model name** - Different models have separate cache files
2. **Model type** - 'debater' or 'judge'
3. **Question index** - Each question has its own cache entry
4. **Temperature** - Ensures consistency with current settings

### Cache Behavior

**On first run** of a question:
1. Check cache for existing result
2. If not found, run direct QA via API
3. Save result to cache
4. Mark as `cached: false` in CSV

**On subsequent runs** of the same question:
1. Check cache for existing result
2. If found and temperature matches, use cached result
3. Skip API call
4. Mark as `cached: true` in CSV

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
- Questions at the same index are the same across runs
- Direct QA with temperature 0.0 is deterministic enough to cache
- Model behavior is consistent for the same question

### Cache Invalidation

Cache is automatically invalidated when:
- Model name changes
- Temperature changes
- Question dataset changes (different question at same index)

### Performance

- **Cache hit**: ~0ms (instant)
- **Cache miss**: Normal API latency + cache write (~100ms)
- **Storage**: ~500 bytes per cached question

### Limitations

- Cache doesn't track the actual question text (only index)
- If you modify the dataset, you should clear the cache
- Cache doesn't sync across machines (local only)

## Best Practices

1. **Leave caching enabled** (default) for normal use
2. **Clear cache** when changing datasets or major model updates
3. **Review `*_cached` columns** in CSV to understand what ran fresh vs cached
4. **Backup cache files** if you want to preserve baselines across machines
5. **Commit cache files to git** if you want team-wide baseline consistency (optional)

