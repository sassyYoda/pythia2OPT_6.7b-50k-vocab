"""
Anchor Quality Analysis (RelRep Only)

Analyzes the quality of anchor tokens used in relative representations:
- Coverage analysis (PCA variance)
- Anchor diversity
- Informativeness (if relative representations are available)
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


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


def analyze_anchor_coverage(anchor_ids, all_token_ids, glove_model, n_components=50):
    """Analyze how well anchors cover the semantic space."""
    # Get embeddings
    anchor_embeddings = []
    for aid in anchor_ids:
        aid_str = str(aid)
        if aid_str in glove_model:
            anchor_embeddings.append(glove_model[aid_str])
    
    all_embeddings = []
    for tid in all_token_ids[:10000]:  # Sample for efficiency
        tid_str = str(tid)
        if tid_str in glove_model:
            all_embeddings.append(glove_model[tid_str])
    
    if len(anchor_embeddings) == 0 or len(all_embeddings) == 0:
        return None, None, None
    
    anchor_embeddings = np.array(anchor_embeddings)
    all_embeddings = np.array(all_embeddings)
    
    # Fit PCA
    anchor_pca = PCA(n_components=min(n_components, len(anchor_embeddings)))
    anchor_pca.fit(anchor_embeddings)
    anchor_variance = anchor_pca.explained_variance_ratio_
    
    full_pca = PCA(n_components=min(n_components, len(all_embeddings)))
    full_pca.fit(all_embeddings)
    full_variance = full_pca.explained_variance_ratio_
    
    # Compute correlation
    min_len = min(len(anchor_variance), len(full_variance))
    coverage_score = np.corrcoef(anchor_variance[:min_len], full_variance[:min_len])[0, 1]
    
    return anchor_variance, full_variance, coverage_score


def analyze_anchor_diversity(anchor_ids, glove_model):
    """Analyze diversity of anchor tokens."""
    anchor_embeddings = []
    for aid in anchor_ids:
        aid_str = str(aid)
        if aid_str in glove_model:
            anchor_embeddings.append(glove_model[aid_str])
    
    if len(anchor_embeddings) < 2:
        return None, None
    
    anchor_embeddings = np.array(anchor_embeddings)
    
    # Compute pairwise distances
    pairwise_dists = pdist(anchor_embeddings)
    mean_distance = np.mean(pairwise_dists)
    std_distance = np.std(pairwise_dists)
    
    return mean_distance, std_distance


def analyze_informativeness(anchor_ids, rel_reps, target_ids):
    """Analyze informativeness of each anchor."""
    if rel_reps is None or len(rel_reps) == 0:
        return None
    
    informativeness = []
    rel_reps_array = np.array(rel_reps)
    
    for anchor_idx in range(len(anchor_ids)):
        if anchor_idx < rel_reps_array.shape[1]:
            similarities_to_anchor = rel_reps_array[:, anchor_idx]
            variance = np.var(similarities_to_anchor)
            informativeness.append(variance)
        else:
            informativeness.append(0.0)
    
    return informativeness


def plot_anchor_quality(anchor_variance, full_variance, informativeness, output_path):
    """Create plots for anchor quality analysis."""
    fig = plt.figure(figsize=(14, 5))
    
    # Plot 1: Scree plot
    ax1 = plt.subplot(1, 2, 1)
    if anchor_variance is not None and full_variance is not None:
        n_components = min(len(anchor_variance), len(full_variance), 20)
        x = np.arange(1, n_components + 1)
        ax1.plot(x, anchor_variance[:n_components], 'o-', label='Anchors', color='#2ecc71', linewidth=2)
        ax1.plot(x, full_variance[:n_components], 's-', label='All Tokens', color='#3498db', linewidth=2)
        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=11)
        ax1.set_title('PCA Variance Explained', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # Plot 2: Informativeness
    ax2 = plt.subplot(1, 2, 2)
    if informativeness is not None:
        top_k = min(20, len(informativeness))
        top_indices = np.argsort(informativeness)[-top_k:][::-1]
        top_values = [informativeness[i] for i in top_indices]
        
        bars = ax2.barh(range(top_k), top_values, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([f'Anchor {i+1}' for i in top_indices], fontsize=9)
        ax2.set_xlabel('Variance (Informativeness)', fontsize=11)
        ax2.set_title(f'Top {top_k} Most Informative Anchors', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            ax2.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anchor quality plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze anchor quality")
    parser.add_argument(
        "--anchor-file",
        type=str,
        default="./data/pythia2opt-6.7b/align_matrix_relrep_anchors.json",
        help="Path to anchor tokens file"
    )
    parser.add_argument(
        "--target-glove",
        type=str,
        default="./data/vec-mix-opt-6.7b.txt",
        help="Path to target GloVe vectors"
    )
    parser.add_argument(
        "--rel-reps-file",
        type=str,
        default=None,
        help="Path to relative representations file (optional)"
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
    print("Anchor Quality Analysis")
    print("="*80)
    
    # Load anchor file
    if not os.path.exists(args.anchor_file):
        print(f"Error: Anchor file not found: {args.anchor_file}")
        print("Please run extract_anchors_relrep.sh first or ensure anchors were saved.")
        return
    
    print(f"\nLoading anchor file: {args.anchor_file}")
    with open(args.anchor_file, 'r') as f:
        anchor_data = json.load(f)
    
    target_anchor_ids = anchor_data.get('target_anchor_ids', [])
    print(f"Found {len(target_anchor_ids)} anchor tokens")
    
    # Load GloVe embeddings
    print("\nLoading GloVe embeddings...")
    target_glove = load_glove_model(args.target_glove)
    
    # Get all token IDs for comparison
    all_token_ids = list(target_glove.keys())[:10000]  # Sample for efficiency
    
    # Analyze coverage
    print("\nAnalyzing anchor coverage...")
    anchor_variance, full_variance, coverage_score = analyze_anchor_coverage(
        target_anchor_ids, all_token_ids, target_glove
    )
    
    if coverage_score is not None:
        print(f"Coverage score (correlation): {coverage_score:.4f}")
    
    # Analyze diversity
    print("\nAnalyzing anchor diversity...")
    mean_dist, std_dist = analyze_anchor_diversity(target_anchor_ids, target_glove)
    
    if mean_dist is not None:
        print(f"Mean pairwise distance: {mean_dist:.4f}")
        print(f"Std pairwise distance: {std_dist:.4f}")
    
    # Analyze informativeness (if available)
    informativeness = None
    if args.rel_reps_file and os.path.exists(args.rel_reps_file):
        print("\nLoading relative representations...")
        rel_reps_data = np.load(args.rel_reps_file, allow_pickle=True).item()
        target_rel_reps = rel_reps_data.get('target_rel_reps', None)
        target_ids = rel_reps_data.get('target_ids', None)
        
        if target_rel_reps is not None:
            print("Analyzing informativeness...")
            informativeness = analyze_informativeness(target_anchor_ids, target_rel_reps, target_ids)
            if informativeness:
                print(f"Mean informativeness: {np.mean(informativeness):.4f}")
    
    # Compile metrics
    metrics = {
        'num_anchors': len(target_anchor_ids),
        'coverage_score': float(coverage_score) if coverage_score is not None else None,
        'mean_pairwise_distance': float(mean_dist) if mean_dist is not None else None,
        'std_pairwise_distance': float(std_dist) if std_dist is not None else None,
        'mean_informativeness': float(np.mean(informativeness)) if informativeness else None
    }
    
    # Save metrics
    output_path = os.path.join(args.output_dir, "anchor_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_anchor_quality(anchor_variance, full_variance, informativeness,
                       os.path.join(args.figures_dir, "anchor_quality.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

