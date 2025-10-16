# Debate Modes: Interactive vs Non-Interactive

## Overview

Each question now runs **TWO** debates:
1. **Interactive Debate** - Judge can ask clarifying questions to specific debaters
2. **Non-Interactive Debate** - Judge can only say "next" or "end"

Both debates are run automatically for every question, and results are saved separately in the CSV.

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

Each row contains results from BOTH debate types:

### Shared Columns:
- `timestamp` - When the debate ran
- `question_idx` - Question ID from dataset
- `debater_direct_correct` - Whether debater got it right without debate
- `debater_confidence` - Debater's confidence
- `judge_direct_correct` - Whether judge got it right without debate  
- `judge_confidence` - Judge's confidence

### Interactive Debate Columns:
- `interactive_debate_turns` - Number of turns in interactive debate
- `interactive_judge_verdict_winner` - Winner (A or B)
- `interactive_judge_after_debate_correct` - Whether judge's verdict was correct
- `interactive_judge_after_debate_confidence` - Judge's confidence in verdict

### Non-Interactive Debate Columns:
- `non_interactive_debate_turns` - Number of turns in non-interactive debate
- `non_interactive_judge_verdict_winner` - Winner (A or B)
- `non_interactive_judge_after_debate_correct` - Whether judge's verdict was correct
- `non_interactive_judge_after_debate_confidence` - Judge's confidence in verdict

## Detail Files

Each debate generates a detailed text file containing:
1. Question details
2. Debater direct QA
3. Judge direct QA
4. **Interactive debate transcript + verdict**
5. **Non-interactive debate transcript + verdict**
6. Summary statistics

## Why Both Modes?

This allows comparison of debate effectiveness:
- Does interaction help the judge make better decisions?
- Do interactive debates take more or fewer turns?
- Does confidence differ between modes?
- Which mode leads to more accurate verdicts?

## Example Usage

```bash
# Run single debate (automatically runs both modes)
python run_single_debate.py --seed 42

# Run parallel debates (each question runs both modes)
python run_parallel_debates.py --num-debates 2 --auto-monitor
```

## Performance Note

Since each question now runs TWO complete debates, execution time is approximately doubled. A typical run that took 30 seconds now takes about 60 seconds.

