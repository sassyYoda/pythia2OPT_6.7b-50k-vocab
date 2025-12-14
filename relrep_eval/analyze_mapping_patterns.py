"""
Many-to-One Mapping Pattern Analysis

Analyzes mapping patterns:
- Distribution of targets per source
- Hub tokens (most-mapped-to sources)
- Gini coefficient for mapping inequality
"""

import json
import os
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def load_alignment_matrix(path):
    """Load alignment matrix from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
    
def compute_gini_coefficient(values):
    """Compute Gini coefficient for inequality measurement."""
    if len(values) == 0:
        return 0.0
    
    values = np.array(sorted(values))
    n = len(values)
    index = np.arange(1, n + 1)
    
    numerator = np.sum((2 * index - n - 1) * values)
    denominator = n * np.sum(values)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def analyze_mapping_patterns(align_matrix, source_tokenizer):
    """Analyze mapping patterns in alignment matrix."""
    # Count how many targets map to each source
    source_counts = defaultdict(int)
    
    for target_id, source_id in align_matrix.items():
        source_counts[source_id] += 1
    
    # Statistics
    counts = list(source_counts.values())
    
    stats = {
        'mean_targets_per_source': np.mean(counts) if counts else 0.0,
        'median_targets_per_source': np.median(counts) if counts else 0.0,
        'std_targets_per_source': np.std(counts) if counts else 0.0,
        'max_targets_per_source': max(counts) if counts else 0,
        'min_targets_per_source': min(counts) if counts else 0,
        'num_unique_sources': len(source_counts),
        'num_unused_sources': 0,  # Will be computed if we know total vocab size
        'gini_coefficient': compute_gini_coefficient(counts)
    }
    
    # Find hub tokens (most-mapped-to sources)
    top_k = 20
    hub_tokens = Counter(source_counts).most_common(top_k)
    
    # Decode hub tokens
    hub_tokens_decoded = []
    for source_id, count in hub_tokens:
        try:
            token_str = source_tokenizer.convert_ids_to_tokens([int(source_id)])[0]
            hub_tokens_decoded.append({
                'source_id': int(source_id),
                'token': token_str,
                'num_targets': count
            })
        except Exception:
            hub_tokens_decoded.append({
                'source_id': int(source_id),
                'token': f'<ID:{source_id}>',
                'num_targets': count
            })
    
    return stats, hub_tokens_decoded, source_counts


def plot_mapping_distribution(source_counts_relrep, source_counts_vanilla, output_path):
    """Plot histogram of mapping count distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts_relrep = list(source_counts_relrep.values())
    counts_vanilla = list(source_counts_vanilla.values())
    
    # Plot 1: RelRep
    ax1 = axes[0]
    ax1.hist(counts_relrep, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Targets per Source', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('RelRep Mapping Distribution', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Vanilla
    ax2 = axes[1]
    ax2.hist(counts_vanilla, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Targets per Source', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Vanilla Mapping Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mapping distribution plot to {output_path}")


def plot_hub_tokens(hub_relrep, hub_vanilla, output_path):
    """Plot bar chart of top hub tokens."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: RelRep hubs
    ax1 = axes[0]
    tokens = [h['token'][:20] for h in hub_relrep[:15]]  # Truncate long tokens
    counts = [h['num_targets'] for h in hub_relrep[:15]]
    
    bars1 = ax1.barh(range(len(tokens)), counts, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(tokens)))
    ax1.set_yticklabels(tokens, fontsize=9)
    ax1.set_xlabel('Number of Targets', fontsize=11)
    ax1.set_title('RelRep Top 15 Hub Tokens', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax1.text(count, i, f' {count}', va='center', fontsize=9)
    
    # Plot 2: Vanilla hubs
    ax2 = axes[1]
    tokens = [h['token'][:20] for h in hub_vanilla[:15]]
    counts = [h['num_targets'] for h in hub_vanilla[:15]]
    
    bars2 = ax2.barh(range(len(tokens)), counts, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels(tokens, fontsize=9)
    ax2.set_xlabel('Number of Targets', fontsize=11)
    ax2.set_title('Vanilla Top 15 Hub Tokens', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars2, counts)):
        ax2.text(count, i, f' {count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hub tokens plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze mapping patterns")
    parser.add_argument(
        "--relrep-matrix",
        type=str,
        default="./data/pythia2opt-6.7b/align_matrix_relrep.json",
        help="Path to relrep alignment matrix"
    )
    parser.add_argument(
        "--vanilla-matrix",
        type=str,
        default="./data/pythia2opt-6.7b/align_matrix.json",
        help="Path to vanilla alignment matrix"
    )
    parser.add_argument(
        "--source-tokenizer",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Source tokenizer path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/pythia2opt-6.7b",
        help="Output directory for results"
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="./figures/relrep-vs-origin",
        help="Directory for figures"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    
    print("="*80)
    print("Many-to-One Mapping Pattern Analysis")
    print("="*80)
    
    # Load data
    print("\nLoading alignment matrices...")
    align_relrep = load_alignment_matrix(args.relrep_matrix)
    align_vanilla = load_alignment_matrix(args.vanilla_matrix)
    
    print("\nLoading tokenizer...")
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, trust_remote_code=True)
    
    # Analyze patterns
    print("\nAnalyzing RelRep mapping patterns...")
    stats_relrep, hub_relrep, source_counts_relrep = analyze_mapping_patterns(
        align_relrep, source_tokenizer
    )
    
    print("\nAnalyzing Vanilla mapping patterns...")
    stats_vanilla, hub_vanilla, source_counts_vanilla = analyze_mapping_patterns(
        align_vanilla, source_tokenizer
    )
    
    # Print statistics
    print("\nRelRep Statistics:")
    for key, value in stats_relrep.items():
        print(f"  {key:30s}: {value}")
    
    print("\nVanilla Statistics:")
    for key, value in stats_vanilla.items():
        print(f"  {key:30s}: {value}")
    
    print("\nRelRep Top 10 Hub Tokens:")
    for i, hub in enumerate(hub_relrep[:10], 1):
        print(f"  {i:2d}. {hub['token']:20s} (ID: {hub['source_id']:6d}) -> {hub['num_targets']:4d} targets")
    
    print("\nVanilla Top 10 Hub Tokens:")
    for i, hub in enumerate(hub_vanilla[:10], 1):
        print(f"  {i:2d}. {hub['token']:20s} (ID: {hub['source_id']:6d}) -> {hub['num_targets']:4d} targets")
    
    # Save metrics
    metrics = {
        'relrep_stats': stats_relrep,
        'vanilla_stats': stats_vanilla,
        'relrep_hub_tokens': hub_relrep,
        'vanilla_hub_tokens': hub_vanilla
    }
    
    output_path = os.path.join(args.output_dir, "mapping_stats.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Save hub tokens as text
    hub_text_path = os.path.join(args.output_dir, "hub_tokens.txt")
    with open(hub_text_path, 'w') as f:
        f.write("RelRep Top 20 Hub Tokens:\n")
        f.write("=" * 60 + "\n")
        for i, hub in enumerate(hub_relrep[:20], 1):
            f.write(f"{i:2d}. {hub['token']:30s} (ID: {hub['source_id']:8d}) -> {hub['num_targets']:4d} targets\n")
        
        f.write("\n\nVanilla Top 20 Hub Tokens:\n")
        f.write("=" * 60 + "\n")
        for i, hub in enumerate(hub_vanilla[:20], 1):
            f.write(f"{i:2d}. {hub['token']:30s} (ID: {hub['source_id']:8d}) -> {hub['num_targets']:4d} targets\n")
    
    print(f"Saved hub tokens to {hub_text_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_mapping_distribution(source_counts_relrep, source_counts_vanilla,
                              os.path.join(args.figures_dir, "mapping_patterns.png"))
    plot_hub_tokens(hub_relrep, hub_vanilla,
                   os.path.join(args.figures_dir, "hub_tokens.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

