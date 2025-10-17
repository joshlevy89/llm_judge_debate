# Debate Modes: Interactive vs Non-Interactive

## Overview

The debate system supports two distinct modes that can be configured via the `DEBATE_MODE` setting in `config.py`:

1. **Interactive Debate** - Judge can ask clarifying questions to specific debaters
2. **Non-Interactive Debate** - Judge can only say "next" or "end"

## Configuration

Set `DEBATE_MODE` in `config.py` to one of:
- `'interactive'` - Run only interactive debates (judge can ask questions)
- `'non_interactive'` - Run only non-interactive debates (judge can only say next/end)
- `'both'` - Run both modes for comparison (**doubles execution time**)

**Default:** `'both'` (for backward compatibility and comparison purposes)

## Interactive Mode

The judge has full control:
- `next` - Let the next debater speak
- `end` - End the debate  
- `A: <question>` - Ask Debater A a specific question
- `B: <question>` - Ask Debater B a specific question

This allows the judge to probe arguments, ask for clarification, and dig deeper into specific claims.

## Non-Interactive Mode

The judge can only:
- `next` - Let the next debater speak
- `end` - End the debate

This is a more passive judging style where debaters alternate and the judge just listens until ready to decide.

## CSV Output Columns

The CSV columns adapt based on which mode(s) you run:

### Shared Columns:
- `timestamp` - When the debate ran
- `question_idx` - Question ID from dataset
- `debater_direct_correct` - Whether debater got it right without debate
- `debater_confidence` - Debater's confidence
- `judge_direct_correct` - Whether judge got it right without debate  
- `judge_confidence` - Judge's confidence

### Interactive Debate Columns (only if `DEBATE_MODE` includes 'interactive'):
- `interactive_debate_turns` - Number of turns in interactive debate
- `interactive_judge_verdict_winner` - Winner (A or B)
- `interactive_judge_after_debate_correct` - Whether judge's verdict was correct
- `interactive_judge_after_debate_confidence` - Judge's confidence in verdict

### Non-Interactive Debate Columns (only if `DEBATE_MODE` includes 'non_interactive'):
- `non_interactive_debate_turns` - Number of turns in non-interactive debate
- `non_interactive_judge_verdict_winner` - Winner (A or B)
- `non_interactive_judge_after_debate_correct` - Whether judge's verdict was correct
- `non_interactive_judge_after_debate_confidence` - Judge's confidence in verdict

**Note:** Columns for modes not run will contain `None` values.

## Detail Files

Each debate generates a detailed text file containing:
1. Question details
2. Debater direct QA
3. Judge direct QA
4. **Interactive debate transcript + verdict** (if `DEBATE_MODE` includes 'interactive')
5. **Non-interactive debate transcript + verdict** (if `DEBATE_MODE` includes 'non_interactive')
6. Summary statistics (adapted to modes run)

## Why Use Different Modes?

**Interactive Mode Benefits:**
- Judge can probe specific claims
- Allows targeted clarification questions
- More thorough exploration of arguments

**Non-Interactive Mode Benefits:**
- Faster execution (no judge question overhead)
- More realistic for some debate scenarios
- Debaters must anticipate judge's questions

**Running Both (`DEBATE_MODE='both'`):**
- Enables direct comparison of effectiveness
- Answers research questions like:
  - Does interaction help the judge make better decisions?
  - Do interactive debates take more or fewer turns?
  - Does confidence differ between modes?
  - Which mode leads to more accurate verdicts?
- **Note:** Doubles execution time since each question runs 2 complete debates

## Example Usage

```bash
# Run single debate with current config mode (set in config.py)
python run_single_debate.py --seed 42

# Run parallel debates with current config mode
python run_parallel_debates.py --num-debates 2 --wait
```

## Changing Modes

To change which mode runs, edit `config.py`:

```python
# Run only interactive mode (faster)
DEBATE_MODE = 'interactive'

# Run only non-interactive mode (faster)
DEBATE_MODE = 'non_interactive'

# Run both modes for comparison (slower, but comprehensive)
DEBATE_MODE = 'both'
```

## Performance Considerations

- **Single mode** (`'interactive'` or `'non_interactive'`): Normal execution time
- **Both modes** (`'both'`): Execution time is approximately **doubled** since each question runs 2 complete debates
  - Example: A run that takes 30 seconds in single mode will take ~60 seconds in 'both' mode

**Recommendation:** Use single mode for fast iteration and `'both'` mode for final comparative analysis.

