# Seeding and Cache Behavior

## Problem (Fixed)

Previously, the cache hit rate was only **16.7%** even when using the same `master_seed` and overlapping questions across runs. This was because two randomizations in `load_gpqa_question()` were not seeded:

1. **Line 207**: `incorrect_answer = random.choice(incorrect_answers)` - randomly selected 1 of 3 incorrect answers
2. **Line 211**: `random.shuffle(positions)` - randomly assigned correct answer to position A or B

Expected cache hit probability: 1/3 × 1/2 = **16.7%**

## Solution

Added `master_seed` parameter to `load_gpqa_question()` that seeds the random number generator before these operations:

```python
if master_seed is not None and question_idx is not None:
    random.seed(master_seed + question_idx)
```

## Result

- ✅ Same `master_seed` + `question_idx` → **100% deterministic** question setup
- ✅ Cache hit rate: **16.7%** → **100%** for overlapping questions
- ✅ Fully reproducible experiments
- ✅ Maximum cache effectiveness

## Usage

When running debates with `run_parallel_debates.py`, the `master_seed` from config is automatically passed through to control:
- Which questions are sampled (without replacement)
- Which incorrect answer is selected for each question
- Which position (A/B) gets the correct answer

## Future: Order Effect Control

For explicit counterbalancing experiments (all correct at A vs all correct at B), an `--order-variant` parameter can be added later to explicitly control position assignment independently of the pseudo-random seed-based assignment.

