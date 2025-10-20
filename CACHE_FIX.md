# Cache Race Condition Fix

## Problem
When running many debates concurrently, cache entries were being lost due to a **race condition** in the baseline cache implementation.

### Root Cause
The `save_qa_to_cache()` function followed an unsafe read-modify-write pattern:
1. Read entire cache file
2. Modify in memory
3. Write back

When multiple processes executed this simultaneously, later writes would overwrite earlier ones, causing cache entries to be lost.

### Why It Only Happened with Many Debates
- With few concurrent debates, timing rarely overlapped
- With many concurrent debates, multiple processes would read the same state before any could write, causing lost updates

## Solution
Added **file locking** using the `filelock` library to ensure atomic read-modify-write operations.

### Changes Made
1. Added `filelock` import to `baseline_cache.py`
2. Modified `save_qa_to_cache()` to acquire an exclusive lock for the entire read-modify-write cycle
3. Updated `_save_cache()` to accept a lock parameter (lock must be held by caller)
4. Created `requirements.txt` with all project dependencies

### Lock Behavior
- Uses a `.lock` file alongside each cache file
- 30-second timeout to prevent deadlocks
- Processes wait in queue for their turn to update the cache
- Ensures no concurrent reads during writes

## Testing
To verify the fix works:
```bash
# Run many debates concurrently and verify all cache entries are saved
python run_parallel_debates.py --num-debates 100 --wait
```

Check that the cache files in `baseline_cache/` contain entries for all unique questions processed.

