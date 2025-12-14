"""
Direct Alignment Quality Metrics - Agreement Analysis

Compares align_matrix_relrep.json with align_matrix.json to analyze:
- Exact agreement between methods
- Agreement with gold mappings
- Visualizations (bar charts, Venn diagrams)
"""

import json
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
try:
    from matplotlib_venn import venn2
except ImportError:
    print("Warning: matplotlib_venn not installed. Install with: pip install matplotlib-venn")
    venn2 = None
import numpy as np


def load_alignment_matrix(path):
    """Load alignment matrix from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_exact_agreement(align_relrep, align_vanilla):
    """Compute exact agreement rate between two alignment matrices."""
    exact_matches = 0
    total = 0
    
    # Get all target IDs (should be same in both)
    target_ids = set(align_relrep.keys()) | set(align_vanilla.keys())
    
    for target_id in target_ids:
        if target_id in align_relrep and target_id in align_vanilla:
            if align_relrep[target_id] == align_vanilla[target_id]:
                exact_matches += 1
            total += 1
    
    return exact_matches, total, exact_matches / total if total > 0 else 0.0


def compute_gold_agreement(align_matrix, gold_mappings):
    """Compute agreement rate with gold mappings."""
    correct = 0
    total = 0
    
    for target_id, gold_source in gold_mappings.items():
        if target_id in align_matrix:
            if align_matrix[target_id] == gold_source:
                correct += 1
            total += 1
    
    return correct, total, correct / total if total > 0 else 0.0


def analyze_agreement_categories(align_relrep, align_vanilla, gold_mappings):
    """Categorize agreements into different types."""
    categories = {
        'both_correct': 0,
        'only_relrep_correct': 0,
        'only_vanilla_correct': 0,
        'both_wrong': 0,
        'both_wrong_but_agree': 0,
        'both_wrong_and_disagree': 0
    }
    
    for target_id, gold_source in gold_mappings.items():
        if target_id not in align_relrep or target_id not in align_vanilla:
            continue
            
        relrep_source = align_relrep[target_id]
        vanilla_source = align_vanilla[target_id]
        
        relrep_correct = (relrep_source == gold_source)
        vanilla_correct = (vanilla_source == gold_source)
        
        if relrep_correct and vanilla_correct:
            categories['both_correct'] += 1
        elif relrep_correct and not vanilla_correct:
            categories['only_relrep_correct'] += 1
        elif not relrep_correct and vanilla_correct:
            categories['only_vanilla_correct'] += 1
        else:  # both wrong
            categories['both_wrong'] += 1
            if relrep_source == vanilla_source:
                categories['both_wrong_but_agree'] += 1
            else:
                categories['both_wrong_and_disagree'] += 1
    
    return categories


def plot_agreement_bar_chart(metrics, output_path):
    """Create bar chart comparing agreement metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Exact Agreement', 'RelRep-Gold', 'Vanilla-Gold']
    values = [
        metrics['exact_agreement_rate'],
        metrics['relrep_gold_agreement_rate'],
        metrics['vanilla_gold_agreement_rate']
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Agreement Rate', fontsize=12)
    ax.set_title('Alignment Agreement Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved agreement bar chart to {output_path}")


def plot_venn_diagram(metrics, output_path):
    """Create Venn diagram showing agreement overlap."""
    if venn2 is None:
        print("Warning: Skipping Venn diagram (matplotlib_venn not available)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate sets
    total_gold = metrics['total_gold_mappings']
    relrep_correct = int(metrics['relrep_gold_agreement_rate'] * total_gold)
    vanilla_correct = int(metrics['vanilla_gold_agreement_rate'] * total_gold)
    both_correct = metrics['agreement_categories']['both_correct']
    
    # Create Venn diagram
    v = venn2(subsets=(vanilla_correct - both_correct, 
                       relrep_correct - both_correct, 
                       both_correct),
              set_labels=('Vanilla Correct', 'RelRep Correct'),
              ax=ax)
    
    # Style
    if v.get_patch_by_id('10'):
        v.get_patch_by_id('10').set_color('#e74c3c')
        v.get_patch_by_id('10').set_alpha(0.5)
    if v.get_patch_by_id('01'):
        v.get_patch_by_id('01').set_color('#2ecc71')
        v.get_patch_by_id('01').set_alpha(0.5)
    if v.get_patch_by_id('11'):
        v.get_patch_by_id('11').set_color('#3498db')
        v.get_patch_by_id('11').set_alpha(0.5)
    
    ax.set_title('Agreement with Gold Mappings', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Venn diagram to {output_path}")


def plot_confusion_matrix(categories, output_path):
    """Create confusion matrix style heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create confusion matrix
    matrix = np.array([
        [categories['both_correct'], categories['only_relrep_correct']],
        [categories['only_vanilla_correct'], categories['both_wrong']]
    ])
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Vanilla\nCorrect', 'Vanilla\nWrong'])
    ax.set_yticklabels(['RelRep\nCorrect', 'RelRep\nWrong'])
    
    ax.set_xlabel('Vanilla Alignment', fontsize=12)
    ax.set_ylabel('RelRep Alignment', fontsize=12)
    ax.set_title('Agreement Confusion Matrix\n(with Gold Mappings)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare alignment matrices")
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
        "--gold-mappings",
        type=str,
        default="./data/Vocab_count/opt-6.7b2pythia.json",
        help="Path to gold mappings"
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
    print("Direct Alignment Quality Metrics - Agreement Analysis")
    print("="*80)
    
    # Load data
    print("\nLoading alignment matrices...")
    align_relrep = load_alignment_matrix(args.relrep_matrix)
    align_vanilla = load_alignment_matrix(args.vanilla_matrix)
    
    print(f"RelRep matrix: {len(align_relrep)} mappings")
    print(f"Vanilla matrix: {len(align_vanilla)} mappings")
    
    print("\nLoading gold mappings...")
    with open(args.gold_mappings, 'r') as f:
        gold_mappings = json.load(f)
    print(f"Gold mappings: {len(gold_mappings)} pairs")
    
    # Compute exact agreement
    print("\nComputing exact agreement...")
    exact_matches, total_exact, exact_rate = compute_exact_agreement(align_relrep, align_vanilla)
    print(f"Exact matches: {exact_matches}/{total_exact} ({exact_rate:.4f})")
    
    # Compute gold agreement
    print("\nComputing gold agreement...")
    relrep_correct, relrep_total, relrep_gold_rate = compute_gold_agreement(align_relrep, gold_mappings)
    vanilla_correct, vanilla_total, vanilla_gold_rate = compute_gold_agreement(align_vanilla, gold_mappings)
    
    print(f"RelRep-Gold: {relrep_correct}/{relrep_total} ({relrep_gold_rate:.4f})")
    print(f"Vanilla-Gold: {vanilla_correct}/{vanilla_total} ({vanilla_gold_rate:.4f})")
    
    # Analyze agreement categories
    print("\nAnalyzing agreement categories...")
    categories = analyze_agreement_categories(align_relrep, align_vanilla, gold_mappings)
    print(f"Both correct: {categories['both_correct']}")
    print(f"Only RelRep correct: {categories['only_relrep_correct']}")
    print(f"Only Vanilla correct: {categories['only_vanilla_correct']}")
    print(f"Both wrong: {categories['both_wrong']}")
    print(f"  - But agree: {categories['both_wrong_but_agree']}")
    print(f"  - And disagree: {categories['both_wrong_and_disagree']}")
    
    # Compile metrics
    metrics = {
        'exact_agreement_rate': exact_rate,
        'exact_matches': exact_matches,
        'total_comparisons': total_exact,
        'relrep_gold_agreement_rate': relrep_gold_rate,
        'relrep_gold_correct': relrep_correct,
        'relrep_gold_total': relrep_total,
        'vanilla_gold_agreement_rate': vanilla_gold_rate,
        'vanilla_gold_correct': vanilla_correct,
        'vanilla_gold_total': vanilla_total,
        'total_gold_mappings': len(gold_mappings),
        'agreement_categories': categories
    }
    
    # Save metrics
    output_path = os.path.join(args.output_dir, "alignment_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_agreement_bar_chart(metrics, os.path.join(args.figures_dir, "agreement_plot.png"))
    plot_venn_diagram(metrics, os.path.join(args.figures_dir, "venn_diagram.png"))
    plot_confusion_matrix(categories, os.path.join(args.figures_dir, "agreement_confusion_matrix.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

