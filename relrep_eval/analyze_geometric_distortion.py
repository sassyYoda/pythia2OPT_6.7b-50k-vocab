"""
Geometric Analysis - Embedding Space Distortion

Analyzes how well geometric structure is preserved after alignment:
- Distance preservation
- Stress metrics
- Distortion distributions
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def load_glove_model(file_path):
    """Load GloVe embeddings from file."""
    glove_model = {}
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    return glove_model


def load_alignment_matrix(path):
    """Load alignment matrix from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def sample_pairs(target_vocab, num_pairs=10000, seed=42):
    """Randomly sample pairs of target tokens."""
    np.random.seed(seed)
    target_ids = list(target_vocab)
    
    if len(target_ids) < 2:
        return []
    
    pairs = []
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(target_ids), size=2, replace=False)
        pairs.append((target_ids[idx1], target_ids[idx2]))
    
    return pairs


def compute_kruskal_stress(original_dists, aligned_dists):
    """Compute Kruskal's stress formula."""
    original_dists = np.array(original_dists)
    aligned_dists = np.array(aligned_dists)
    
    numerator = np.sum((original_dists - aligned_dists) ** 2)
    denominator = np.sum(original_dists ** 2)
    
    if denominator == 0:
        return 0.0
    
    return np.sqrt(numerator / denominator)


def analyze_geometric_distortion(align_relrep, align_vanilla, source_glove, target_glove, num_pairs=10000):
    """Analyze geometric distortion."""
    # Get all target IDs that have embeddings
    target_ids = []
    for target_id in set(align_relrep.keys()) | set(align_vanilla.keys()):
        if str(target_id) in target_glove:
            target_ids.append(str(target_id))
    
    # Sample pairs
    pairs = sample_pairs(target_ids, num_pairs=min(num_pairs, len(target_ids) * 10))
    
    relrep_distortions = []
    vanilla_distortions = []
    original_dists = []
    relrep_dists = []
    vanilla_dists = []
    
    valid_pairs = 0
    
    for target_a, target_b in pairs:
        # Check if both tokens have embeddings and alignments
        if (target_a not in target_glove or target_b not in target_glove or
            target_a not in align_relrep or target_b not in align_relrep or
            target_a not in align_vanilla or target_b not in align_vanilla):
            continue
        
        # Original distance in target space
        orig_dist = euclidean(target_glove[target_a], target_glove[target_b])
        original_dists.append(orig_dist)
        
        # RelRep aligned distance
        source_a_relrep = str(align_relrep[target_a])
        source_b_relrep = str(align_relrep[target_b])
        
        if source_a_relrep in source_glove and source_b_relrep in source_glove:
            relrep_dist = euclidean(source_glove[source_a_relrep], source_glove[source_b_relrep])
            relrep_dists.append(relrep_dist)
            relrep_distortions.append(abs(orig_dist - relrep_dist))
        else:
            continue
        
        # Vanilla aligned distance
        source_a_vanilla = str(align_vanilla[target_a])
        source_b_vanilla = str(align_vanilla[target_b])
        
        if source_a_vanilla in source_glove and source_b_vanilla in source_glove:
            vanilla_dist = euclidean(source_glove[source_a_vanilla], source_glove[source_b_vanilla])
            vanilla_dists.append(vanilla_dist)
            vanilla_distortions.append(abs(orig_dist - vanilla_dist))
        else:
            continue
        
        valid_pairs += 1
    
    # Compute statistics
    stats = {
        'num_pairs_analyzed': valid_pairs,
        'mean_relrep_distortion': float(np.mean(relrep_distortions)) if relrep_distortions else 0.0,
        'mean_vanilla_distortion': float(np.mean(vanilla_distortions)) if vanilla_distortions else 0.0,
        'median_relrep_distortion': float(np.median(relrep_distortions)) if relrep_distortions else 0.0,
        'median_vanilla_distortion': float(np.median(vanilla_distortions)) if vanilla_distortions else 0.0,
        'std_relrep_distortion': float(np.std(relrep_distortions)) if relrep_distortions else 0.0,
        'std_vanilla_distortion': float(np.std(vanilla_distortions)) if vanilla_distortions else 0.0,
        'kruskal_stress_relrep': compute_kruskal_stress(original_dists[:len(relrep_dists)], relrep_dists),
        'kruskal_stress_vanilla': compute_kruskal_stress(original_dists[:len(vanilla_dists)], vanilla_dists)
    }
    
    return stats, original_dists, relrep_dists, vanilla_dists, relrep_distortions, vanilla_distortions


def plot_geometric_distortion(original_dists, relrep_dists, vanilla_dists, 
                              relrep_distortions, vanilla_distortions, output_path):
    """Create scatter plots and histograms of geometric distortion."""
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Scatter - RelRep
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(original_dists[:len(relrep_dists)], relrep_dists, alpha=0.3, s=10, color='#2ecc71')
    min_val = min(min(original_dists[:len(relrep_dists)]), min(relrep_dists))
    max_val = max(max(original_dists[:len(relrep_dists)]), max(relrep_dists))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Preservation')
    ax1.set_xlabel('Original Distance', fontsize=10)
    ax1.set_ylabel('Aligned Distance', fontsize=10)
    ax1.set_title('RelRep Distance Preservation', fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot 2: Scatter - Vanilla
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(original_dists[:len(vanilla_dists)], vanilla_dists, alpha=0.3, s=10, color='#e74c3c')
    min_val = min(min(original_dists[:len(vanilla_dists)]), min(vanilla_dists))
    max_val = max(max(original_dists[:len(vanilla_dists)]), max(vanilla_dists))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Preservation')
    ax2.set_xlabel('Original Distance', fontsize=10)
    ax2.set_ylabel('Aligned Distance', fontsize=10)
    ax2.set_title('Vanilla Distance Preservation', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Plot 3: Histogram - Distortion comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(relrep_distortions, bins=50, alpha=0.6, color='#2ecc71', label='RelRep', edgecolor='black')
    ax3.hist(vanilla_distortions, bins=50, alpha=0.6, color='#e74c3c', label='Vanilla', edgecolor='black')
    ax3.set_xlabel('Distortion (|Original - Aligned|)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Distortion Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Box plot comparison
    ax4 = plt.subplot(2, 3, 4)
    bp = ax4.boxplot([relrep_distortions, vanilla_distortions], 
                     labels=['RelRep', 'Vanilla'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#e74c3c')
    bp['boxes'][1].set_alpha(0.7)
    ax4.set_ylabel('Distortion', fontsize=10)
    ax4.set_title('Distortion Comparison', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Correlation plot
    ax5 = plt.subplot(2, 3, 5)
    if len(relrep_distortions) == len(vanilla_distortions):
        ax5.scatter(relrep_distortions, vanilla_distortions, alpha=0.3, s=10, color='purple')
        min_val = min(min(relrep_distortions), min(vanilla_distortions))
        max_val = max(max(relrep_distortions), max(vanilla_distortions))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax5.set_xlabel('RelRep Distortion', fontsize=10)
        ax5.set_ylabel('Vanilla Distortion', fontsize=10)
        ax5.set_title('Distortion Correlation', fontsize=11, fontweight='bold')
        ax5.grid(alpha=0.3)
    
    # Plot 6: Summary statistics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = (
        f"Geometric Distortion Statistics\n"
        f"{'='*40}\n\n"
        f"Pairs Analyzed: {len(relrep_distortions):,}\n\n"
        f"RelRep:\n"
        f"  Mean: {np.mean(relrep_distortions):.4f}\n"
        f"  Median: {np.median(relrep_distortions):.4f}\n"
        f"  Std: {np.std(relrep_distortions):.4f}\n\n"
        f"Vanilla:\n"
        f"  Mean: {np.mean(vanilla_distortions):.4f}\n"
        f"  Median: {np.median(vanilla_distortions):.4f}\n"
        f"  Std: {np.std(vanilla_distortions):.4f}"
    )
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved geometric distortion plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze geometric distortion")
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
        "--source-glove",
        type=str,
        default="./data/vec-mix-pythia.txt",
        help="Path to source GloVe vectors"
    )
    parser.add_argument(
        "--target-glove",
        type=str,
        default="./data/vec-mix-opt-6.7b.txt",
        help="Path to target GloVe vectors"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10000,
        help="Number of pairs to sample for analysis"
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
    print("Geometric Analysis - Embedding Space Distortion")
    print("="*80)
    
    # Load data
    print("\nLoading alignment matrices...")
    align_relrep = load_alignment_matrix(args.relrep_matrix)
    align_vanilla = load_alignment_matrix(args.vanilla_matrix)
    
    print("\nLoading GloVe embeddings...")
    source_glove = load_glove_model(args.source_glove)
    target_glove = load_glove_model(args.target_glove)
    
    # Analyze distortion
    print(f"\nAnalyzing geometric distortion (sampling {args.num_pairs} pairs)...")
    stats, orig_dists, relrep_dists, vanilla_dists, relrep_distortions, vanilla_distortions = \
        analyze_geometric_distortion(
            align_relrep, align_vanilla, source_glove, target_glove, args.num_pairs
        )
    
    print(f"\nAnalyzed {stats['num_pairs_analyzed']:,} valid pairs")
    print(f"\nRelRep Distortion:")
    print(f"  Mean: {stats['mean_relrep_distortion']:.4f}")
    print(f"  Median: {stats['median_relrep_distortion']:.4f}")
    print(f"  Std: {stats['std_relrep_distortion']:.4f}")
    print(f"  Kruskal Stress: {stats['kruskal_stress_relrep']:.4f}")
    
    print(f"\nVanilla Distortion:")
    print(f"  Mean: {stats['mean_vanilla_distortion']:.4f}")
    print(f"  Median: {stats['median_vanilla_distortion']:.4f}")
    print(f"  Std: {stats['std_vanilla_distortion']:.4f}")
    print(f"  Kruskal Stress: {stats['kruskal_stress_vanilla']:.4f}")
    
    # Save metrics
    output_path = os.path.join(args.output_dir, "distortion_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Create visualization
    if stats['num_pairs_analyzed'] > 0:
        print("\nCreating visualization...")
        plot_geometric_distortion(orig_dists, relrep_dists, vanilla_dists,
                                 relrep_distortions, vanilla_distortions,
                                 os.path.join(args.figures_dir, "geometric_distortion.png"))
    else:
        print("\nWarning: No valid pairs found for visualization")
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

