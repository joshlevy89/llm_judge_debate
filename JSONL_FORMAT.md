# JSONL Results Format

The debate system now uses JSONL (JSON Lines) format for storing results instead of CSV. This provides:

- **Flexible schema**: Different runs can have different debate modes without empty columns
- **Rich structure**: Nested data for cleaner organization
- **Easy analysis**: Works with pandas, jq, and standard JSON tools
- **Extensible**: Add new modes or fields without breaking existing data

## File Structure

Each run creates a directory with:
```
parallel_debate_runs/
  master_results_seed42_n100_20251017_114016/
    ├── master_results.jsonl      # One JSON object per line
    ├── log_*.txt                  # Detailed logs for each debate
    └── config_used.json           # Configuration snapshot (JSON format)
```

## JSONL Format

Each line is a complete JSON object representing one debate:

```json
{
    "run_id": "3a8c9032",
    "timestamp": "20251017_114044",
    "question_idx": 176,
    "debater_direct": {
        "correct": false,
        "confidence": 90,
        "cached": true
    },
    "judge_direct": {
        "correct": false,
        "confidence": 90,
        "cached": true
    },
    "modes": {
        "non_interactive": {
            "turns": 3,
            "winner": "A",
            "correct": false,
            "confidence": 85
        }
    }
}
```

### Fields

- **run_id**: Unique identifier for this debate
- **timestamp**: When the debate completed
- **question_idx**: Question index from the dataset
- **debater_direct**: Results from debater's direct QA (no debate)
  - `correct`: Whether the answer was correct
  - `confidence`: Confidence level (0-100)
  - `cached`: Whether result was from cache
- **judge_direct**: Results from judge's direct QA (no debate)
  - Same structure as debater_direct
- **modes**: Dictionary of debate mode results
  - Key is mode name: `interactive`, `non_interactive`, or any custom mode
  - Each mode contains:
    - `turns`: Number of debate turns
    - `winner`: Which debater won ('A' or 'B')
    - `correct`: Whether the verdict was correct
    - `confidence`: Judge's confidence in verdict

## Analysis Tools

### Basic Analysis

Use the provided `analyze_jsonl.py` helper:

```bash
# Print summary statistics
python analyze_jsonl.py parallel_debate_runs/master_results_seed42_n100_20251017_114016/master_results.jsonl
```

### Python Analysis

```python
from analyze_jsonl import load_results, results_to_dataframe, compare_modes

# Load results
results = load_results('path/to/master_results.jsonl')

# Convert to pandas DataFrame
df = results_to_dataframe(results)

# Filter by mode
interactive_only = [r for r in results if 'interactive' in r['modes']]

# Compare modes within same run
comparison = compare_modes(results, 'interactive', 'non_interactive')
```

### Command-line Tools

```bash
# Count total debates
wc -l master_results.jsonl

# Extract specific field with jq
jq '.debater_direct.correct' master_results.jsonl

# Filter by mode
jq 'select(.modes | has("interactive"))' master_results.jsonl

# Calculate accuracy
jq -s 'map(select(.modes.non_interactive.correct == true)) | length' master_results.jsonl
```

## Converting Existing CSV Files

If you have old CSV results, convert them with:

```bash
# Convert a single file
python convert_csv_to_jsonl.py path/to/master_results.csv

# Convert all CSV files in a directory
python convert_csv_to_jsonl.py parallel_debate_runs/
```

## Comparing Runs

```python
from analyze_jsonl import compare_runs

# Compare two runs
stats = compare_runs(
    'run1/master_results.jsonl',
    'run2/master_results.jsonl',
    mode='interactive'  # Optional: specify mode to compare
)

print(stats)
# {
#   'run1_total': 100,
#   'run2_total': 100,
#   'interactive_run1_accuracy': 0.52,
#   'interactive_run2_accuracy': 0.48
# }
```

## Example Analysis Workflows

### Compare Interactive vs Non-Interactive

```python
import pandas as pd
from analyze_jsonl import load_results, compare_modes

results = load_results('master_results.jsonl')
comparison = compare_modes(results, 'interactive', 'non_interactive')

# Show disagreements
disagreements = comparison[comparison['agreement'] == False]
print(f"Disagreements: {len(disagreements)} / {len(comparison)}")
```

### Analyze by Confidence

```python
import pandas as pd
from analyze_jsonl import load_results, results_to_dataframe

results = load_results('master_results.jsonl')
df = results_to_dataframe(results)

# Judge confidence analysis
high_conf = df[df['judge_direct_confidence'] >= 90]
low_conf = df[df['judge_direct_confidence'] < 90]

print(f"High confidence accuracy: {high_conf['judge_direct_correct'].mean():.2%}")
print(f"Low confidence accuracy: {low_conf['judge_direct_correct'].mean():.2%}")
```

### Track Cache Usage

```python
from analyze_jsonl import load_results

results = load_results('master_results.jsonl')

# Count cached vs non-cached
debater_cached = sum(1 for r in results if r['debater_direct']['cached'])
judge_cached = sum(1 for r in results if r['judge_direct']['cached'])

print(f"Debater cache hits: {debater_cached}/{len(results)}")
print(f"Judge cache hits: {judge_cached}/{len(results)}")
```

## Benefits Over CSV

1. **No empty columns**: Only save data for modes that were actually run
2. **Easy filtering**: Use Python dict operations or jq to filter
3. **Rich nested data**: Group related fields naturally
4. **Future-proof**: Add new modes without changing existing files
5. **Standard format**: Works with any JSON tool
6. **Human-readable**: Still easy to inspect with text tools

## Migration Notes

- The CSV format is completely replaced with JSONL
- Use `convert_csv_to_jsonl.py` to migrate old data
- All new runs automatically use JSONL
- The parallel runner aggregates stats from JSONL

