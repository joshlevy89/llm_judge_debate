"""
Check for near-duplicate questions in generated dataset.

Uses rapidfuzz for fast similarity computation to find question pairs
with similarity above a threshold.
"""

import json
import argparse
from rapidfuzz import fuzz
from rapidfuzz.process import cdist


def normalize_text(text):
    """Light normalization: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def find_similar_pairs(questions, threshold=0.95, normalize=True, lower_bound=None, verbose=True):
    """Find all pairs of questions with similarity >= threshold (or in range if lower_bound set)."""
    texts = [normalize_text(q["question"]) if normalize else q["question"] for q in questions]
    n = len(texts)
    total_pairs = n * (n - 1) // 2
    
    if verbose:
        print(f"Computing {total_pairs:,} pairwise similarities...")
    
    # Compute full similarity matrix (values 0-100)
    sim_matrix = cdist(texts, texts, scorer=fuzz.ratio, workers=-1)
    
    # Find pairs in range (convert to 0-100 scale)
    threshold_100 = threshold * 100
    lower_100 = (lower_bound * 100) if lower_bound else 0
    similar_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim >= threshold_100:
                similar_pairs.append((i, j, sim / 100))
            elif lower_bound and sim >= lower_100:
                similar_pairs.append((i, j, sim / 100))
    
    return similar_pairs


def get_duplicate_clusters(pairs, threshold=0.9999):
    """Group duplicate pairs into clusters of equivalent questions."""
    from collections import defaultdict
    
    matching_pairs = [(i, j) for i, j, r in pairs if r >= threshold]
    
    if not matching_pairs:
        return []
    
    # Union-find to cluster
    parent = {}
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j in matching_pairs:
        union(i, j)
    
    clusters = defaultdict(set)
    for i, j in matching_pairs:
        clusters[find(i)].add(i)
        clusters[find(i)].add(j)
    
    return [sorted(c) for c in clusters.values() if len(c) > 1]


def main():
    parser = argparse.ArgumentParser(description="Find near-duplicate questions")
    parser.add_argument("data_file", help="Path to augmented_math JSON file")
    parser.add_argument("--threshold", type=float, default=0.95, help="Similarity threshold (default: 0.95)")
    parser.add_argument("--no-normalize", action="store_true", help="Skip text normalization")
    parser.add_argument("--show-all", action="store_true", help="Show all pairs (default: first 20)")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary stats")
    parser.add_argument("--below", action="store_true", help="Show pairs just below threshold (to check borderline cases)")
    parser.add_argument("--below-range", type=float, default=0.05, help="How far below threshold to look (default: 0.05)")
    parser.add_argument("--all", action="store_true", help="Check all questions (default: only gt_assigned, matching generate_questions.py)")
    args = parser.parse_args()
    
    with open(args.data_file) as f:
        data = json.load(f)
    
    all_questions = data["questions"]
    
    if args.all:
        questions = [q for q in all_questions if q.get("question")]
        filter_desc = "all with text"
    else:
        questions = [q for q in all_questions if q.get("gt_assigned")]
        filter_desc = "gt_assigned only"
    
    print(f"Total in file: {len(all_questions)}")
    print(f"Checking: {len(questions)} ({filter_desc})")
    
    lower_bound = (args.threshold - args.below_range) if args.below else None
    pairs = find_similar_pairs(questions, args.threshold, not args.no_normalize, lower_bound)
    pairs.sort(key=lambda x: -x[2])
    
    # Split by threshold
    above_threshold = [(i, j, r) for i, j, r in pairs if r >= args.threshold]
    below_threshold = [(i, j, r) for i, j, r in pairs if r < args.threshold]
    
    clusters = get_duplicate_clusters(pairs, threshold=args.threshold)
    
    # Determine which indices to keep (first in each cluster, plus all non-clustered)
    removed_indices = set()
    for cluster in clusters:
        removed_indices.update(cluster[1:])  # Keep first, remove rest
    
    kept_indices = set(range(len(questions))) - removed_indices
    num_duplicates = len(removed_indices)
    
    print(f"Found {len(clusters)} duplicate clusters, {num_duplicates} questions marked as duplicates")
    print(f"Valid: {len(kept_indices)}")
    
    if args.below:
        print(f"\nNear-miss pairs ({(args.threshold-args.below_range)*100:.0f}%-{args.threshold*100:.0f}%): {len(below_threshold)}")
    
    if args.summary_only:
        return
    
    if args.below:
        # Filter to pairs where BOTH questions are in the kept set (post-dedup near-misses)
        below_kept = [(i, j, r) for i, j, r in below_threshold if i in kept_indices and j in kept_indices]
        display_pairs = below_kept
        label = f"near-miss pairs (post-dedup)"
        print(f"\nNear-miss pairs among {len(kept_indices)} valid questions: {len(below_kept)}")
    else:
        display_pairs = above_threshold
        label = f"duplicate pairs"
    
    if not display_pairs:
        return
    
    show_count = len(display_pairs) if args.show_all else min(20, len(display_pairs))
    print(f"\nShowing {show_count} {label}:\n")
    
    for idx, (i, j, ratio) in enumerate(display_pairs[:show_count], 1):
        q1, q2 = questions[i], questions[j]
        print(f"{'='*60}")
        print(f"Pair {idx}: similarity = {ratio:.4f}")
        print(f"  Q{i} (L{q1['level']}/{q1['subject']}):")
        print(f"    {q1['question'][:200]}{'...' if len(q1['question']) > 200 else ''}")
        print(f"  Q{j} (L{q2['level']}/{q2['subject']}):")
        print(f"    {q2['question'][:200]}{'...' if len(q2['question']) > 200 else ''}")
    
    if len(display_pairs) > show_count:
        print(f"\n... and {len(display_pairs) - show_count} more (use --show-all to see all)")


if __name__ == "__main__":
    main()
