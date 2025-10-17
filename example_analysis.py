#!/usr/bin/env python3
"""
Example analysis scenarios with JSONL format.

This demonstrates the flexibility of the new format for common analysis tasks.
"""

from analyze_jsonl import load_results, results_to_dataframe, compare_modes, filter_by_mode
import pandas as pd


def example_1_compare_modes_same_run():
    """
    Compare interactive vs non-interactive from the same run.
    """
    print("="*70)
    print("Example 1: Compare modes within the same run")
    print("="*70)
    
    # Load results that have both modes
    results = load_results('run1/master_results.jsonl')
    
    # Get only debates that ran both modes
    both_modes = [r for r in results 
                  if 'interactive' in r['modes'] and 'non_interactive' in r['modes']]
    
    print(f"Found {len(both_modes)} debates with both modes\n")
    
    # Compare accuracy
    interactive_correct = sum(1 for r in both_modes if r['modes']['interactive']['correct'])
    non_interactive_correct = sum(1 for r in both_modes if r['modes']['non_interactive']['correct'])
    
    print(f"Interactive accuracy: {interactive_correct}/{len(both_modes)} ({100*interactive_correct/len(both_modes):.1f}%)")
    print(f"Non-interactive accuracy: {non_interactive_correct}/{len(both_modes)} ({100*non_interactive_correct/len(both_modes):.1f}%)")
    
    # Show where they disagree
    disagreements = [r for r in both_modes 
                     if r['modes']['interactive']['correct'] != r['modes']['non_interactive']['correct']]
    
    print(f"\nDisagreements: {len(disagreements)} cases")
    print("\nQuestion indices where modes disagreed:")
    for r in disagreements[:5]:  # Show first 5
        int_correct = r['modes']['interactive']['correct']
        non_int_correct = r['modes']['non_interactive']['correct']
        print(f"  Q{r['question_idx']}: interactive={int_correct}, non_interactive={non_int_correct}")


def example_2_compare_across_runs():
    """
    Get interactive from run A and compare to non-interactive from run B.
    """
    print("\n" + "="*70)
    print("Example 2: Compare modes across different runs")
    print("="*70)
    
    # Load two different runs
    run_a = load_results('run_a/master_results.jsonl')
    run_b = load_results('run_b/master_results.jsonl')
    
    # Extract specific modes
    run_a_interactive = filter_by_mode(run_a, 'interactive')
    run_b_non_interactive = filter_by_mode(run_b, 'non_interactive')
    
    print(f"Run A (interactive): {len(run_a_interactive)} debates")
    print(f"Run B (non-interactive): {len(run_b_non_interactive)} debates\n")
    
    # Compare accuracy
    a_correct = sum(1 for r in run_a_interactive if r['modes']['interactive']['correct'])
    b_correct = sum(1 for r in run_b_non_interactive if r['modes']['non_interactive']['correct'])
    
    print(f"Run A interactive accuracy: {100*a_correct/len(run_a_interactive):.1f}%")
    print(f"Run B non-interactive accuracy: {100*b_correct/len(run_b_non_interactive):.1f}%")
    
    # If they used same questions, can do question-by-question comparison
    a_by_question = {r['question_idx']: r['modes']['interactive']['correct'] 
                     for r in run_a_interactive}
    b_by_question = {r['question_idx']: r['modes']['non_interactive']['correct'] 
                     for r in run_b_non_interactive}
    
    common_questions = set(a_by_question.keys()) & set(b_by_question.keys())
    if common_questions:
        print(f"\n{len(common_questions)} questions in common")
        agreements = sum(1 for q in common_questions if a_by_question[q] == b_by_question[q])
        print(f"Agreement rate: {100*agreements/len(common_questions):.1f}%")


def example_3_analyze_new_mode():
    """
    Analyze a hypothetical new mode you add in the future.
    """
    print("\n" + "="*70)
    print("Example 3: Analyze a new mode (future-proof)")
    print("="*70)
    
    results = load_results('experimental_run/master_results.jsonl')
    
    # Check what modes are available
    all_modes = set()
    for r in results:
        all_modes.update(r.get('modes', {}).keys())
    
    print(f"Available modes: {', '.join(sorted(all_modes))}\n")
    
    # Maybe you added a new "multi_judge" mode?
    if 'multi_judge' in all_modes:
        multi_judge_results = filter_by_mode(results, 'multi_judge')
        correct = sum(1 for r in multi_judge_results if r['modes']['multi_judge']['correct'])
        print(f"Multi-judge mode:")
        print(f"  Debates: {len(multi_judge_results)}")
        print(f"  Accuracy: {100*correct/len(multi_judge_results):.1f}%")
        
        # Compare to other modes
        for mode in sorted(all_modes - {'multi_judge'}):
            mode_results = filter_by_mode(results, mode)
            mode_correct = sum(1 for r in mode_results if r['modes'][mode]['correct'])
            print(f"\n{mode} comparison:")
            print(f"  Accuracy: {100*mode_correct/len(mode_results):.1f}%")


def example_4_pandas_analysis():
    """
    Use pandas for complex analysis.
    """
    print("\n" + "="*70)
    print("Example 4: Advanced analysis with pandas")
    print("="*70)
    
    results = load_results('run1/master_results.jsonl')
    df = results_to_dataframe(results)
    
    print("\nDataFrame shape:", df.shape)
    print("\nColumns:", list(df.columns))
    
    # Analyze by confidence levels
    if 'non_interactive_confidence' in df.columns:
        print("\nAccuracy by confidence level (non-interactive):")
        
        # Group by confidence ranges
        df['confidence_range'] = pd.cut(df['non_interactive_confidence'], 
                                       bins=[0, 70, 80, 90, 100],
                                       labels=['0-70', '70-80', '80-90', '90-100'])
        
        by_confidence = df.groupby('confidence_range')['non_interactive_correct'].agg(['mean', 'count'])
        print(by_confidence)
    
    # Analyze judge improvement
    print("\nJudge performance before and after debate:")
    judge_improved = df['non_interactive_correct'] & ~df['judge_direct_correct']
    judge_worsened = ~df['non_interactive_correct'] & df['judge_direct_correct']
    
    print(f"  Improved: {judge_improved.sum()} cases")
    print(f"  Worsened: {judge_worsened.sum()} cases")
    print(f"  Unchanged: {(~judge_improved & ~judge_worsened).sum()} cases")


def example_5_mix_and_match():
    """
    Complex scenario: Compare specific mode from multiple runs.
    """
    print("\n" + "="*70)
    print("Example 5: Compare same mode across multiple runs")
    print("="*70)
    
    # Load multiple runs
    runs = {
        'baseline': load_results('baseline_run/master_results.jsonl'),
        'experiment_a': load_results('experiment_a/master_results.jsonl'),
        'experiment_b': load_results('experiment_b/master_results.jsonl'),
    }
    
    mode_to_compare = 'non_interactive'
    
    print(f"Comparing '{mode_to_compare}' mode across runs:\n")
    
    for run_name, results in runs.items():
        mode_results = filter_by_mode(results, mode_to_compare)
        if mode_results:
            correct = sum(1 for r in mode_results if r['modes'][mode_to_compare]['correct'])
            avg_turns = sum(r['modes'][mode_to_compare]['turns'] for r in mode_results) / len(mode_results)
            
            print(f"{run_name}:")
            print(f"  Debates: {len(mode_results)}")
            print(f"  Accuracy: {100*correct/len(mode_results):.1f}%")
            print(f"  Avg turns: {avg_turns:.1f}")
            print()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("JSONL Format - Analysis Examples")
    print("="*70)
    print("\nThese examples show how easy it is to:")
    print("  1. Compare modes within the same run")
    print("  2. Compare modes across different runs")
    print("  3. Analyze new modes you haven't even created yet")
    print("  4. Use pandas for complex analysis")
    print("  5. Mix and match data from multiple runs")
    print("\nThe flexible JSONL format makes all of this simple!")
    print("="*70 + "\n")
    
    # Uncomment to run examples (requires actual data files):
    # example_1_compare_modes_same_run()
    # example_2_compare_across_runs()
    # example_3_analyze_new_mode()
    # example_4_pandas_analysis()
    # example_5_mix_and_match()
    
    print("\nTo run these examples, uncomment the function calls and provide")
    print("paths to your actual JSONL files.")

