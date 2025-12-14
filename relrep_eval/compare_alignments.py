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


def get_non_gold_tokens(align_relrep, align_vanilla, gold_mappings):
    """Get tokens that are NOT in gold mappings (i.e., aligned by methodology)."""
    gold_target_ids = set(gold_mappings.keys())
    all_target_ids = set(align_relrep.keys()) | set(align_vanilla.keys())
    non_gold_ids = all_target_ids - gold_target_ids
    return non_gold_ids


def compute_exact_agreement(align_relrep, align_vanilla, gold_mappings):
    """Compute exact agreement rate between two alignment matrices for NON-GOLD tokens only."""
    non_gold_ids = get_non_gold_tokens(align_relrep, align_vanilla, gold_mappings)
    
    exact_matches = 0
    total = 0
    
    for target_id in non_gold_ids:
        if target_id in align_relrep and target_id in align_vanilla:
            if align_relrep[target_id] == align_vanilla[target_id]:
                exact_matches += 1
            total += 1
    
    return exact_matches, total, exact_matches / total if total > 0 else 0.0


def compute_gold_agreement(align_matrix, gold_mappings):
    """Compute agreement rate with gold mappings (for verification that gold mappings are preserved)."""
    correct = 0
    total = 0
    
    for target_id, gold_source in gold_mappings.items():
        if target_id in align_matrix:
            if align_matrix[target_id] == gold_source:
                correct += 1
            total += 1
    
    return correct, total, correct / total if total > 0 else 0.0


def analyze_agreement_categories(align_relrep, align_vanilla, gold_mappings):
    """Categorize agreements for NON-GOLD tokens only."""
    non_gold_ids = get_non_gold_tokens(align_relrep, align_vanilla, gold_mappings)
    
    categories = {
        'agree': 0,  # Both methods agree
        'disagree': 0,  # Methods disagree
        'total': 0
    }
    
    for target_id in non_gold_ids:
        if target_id not in align_relrep or target_id not in align_vanilla:
            continue
        
        relrep_source = align_relrep[target_id]
        vanilla_source = align_vanilla[target_id]
        
        if relrep_source == vanilla_source:
            categories['agree'] += 1
        else:
            categories['disagree'] += 1
        categories['total'] += 1
    
    return categories


def plot_agreement_bar_chart(metrics, output_path):
    """Create bar chart comparing agreement metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Main comparison: Non-gold tokens
    categories = ['Method Agreement\n(Non-Gold)', 'RelRep-Gold\n(Verification)', 'Vanilla-Gold\n(Verification)']
    values = [
        metrics['exact_agreement_rate'],
        metrics['relrep_gold_agreement_rate'],
        metrics['vanilla_gold_agreement_rate']
    ]
    
    colors = ['#3498db', '#95a5a6', '#95a5a6']  # Highlight the main metric
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Agreement Rate', fontsize=12)
    ax.set_title('Alignment Agreement Comparison\n(Non-Gold Tokens Only)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation
    ax.text(0.5, 0.02, f"Non-Gold Tokens: {metrics['non_gold_total']:,}",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved agreement bar chart to {output_path}")


def plot_agreement_pie_chart(categories, output_path):
    """Create pie chart showing agreement vs disagreement for non-gold tokens."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['Agree', 'Disagree']
    sizes = [categories['agree'], categories['disagree']]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)  # Slight separation
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('RelRep vs Vanilla Agreement\n(Non-Gold Tokens Only)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add count annotation
    total = categories['total']
    ax.text(0, -1.3, f'Total Non-Gold Tokens: {total:,}',
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved agreement pie chart to {output_path}")


def plot_venn_diagram(metrics, output_path):
    """Create Venn diagram showing agreement overlap."""
    if venn2 is None:
        print("Warning: Skipping Venn diagram (matplotlib_venn not available)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate sets (for gold mapping verification)
    total_gold = metrics['total_gold_mappings']
    relrep_correct = metrics['relrep_gold_correct']
    vanilla_correct = metrics['vanilla_gold_correct']
    # Both should be correct (gold mappings are pre-populated)
    both_correct = min(relrep_correct, vanilla_correct)
    
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
    
    ax.set_title('Gold Mapping Preservation\n(Verification Only)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Venn diagram to {output_path}")


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
    
    # Get non-gold tokens (tokens aligned by methodology)
    non_gold_ids = get_non_gold_tokens(align_relrep, align_vanilla, gold_mappings)
    print(f"\nTotal tokens: {len(set(align_relrep.keys()) | set(align_vanilla.keys())):,}")
    print(f"Gold-mapped tokens: {len(gold_mappings):,}")
    print(f"Non-gold tokens (aligned by methodology): {len(non_gold_ids):,}")
    
    # Compute exact agreement for NON-GOLD tokens only
    print("\nComputing exact agreement (NON-GOLD tokens only)...")
    exact_matches, total_exact, exact_rate = compute_exact_agreement(align_relrep, align_vanilla, gold_mappings)
    print(f"Exact matches: {exact_matches:,}/{total_exact:,} ({exact_rate:.4f})")
    
    # Verify gold mappings are preserved (should be 100% or close)
    print("\nVerifying gold mappings are preserved...")
    relrep_correct, relrep_total, relrep_gold_rate = compute_gold_agreement(align_relrep, gold_mappings)
    vanilla_correct, vanilla_total, vanilla_gold_rate = compute_gold_agreement(align_vanilla, gold_mappings)
    
    print(f"RelRep-Gold: {relrep_correct:,}/{relrep_total:,} ({relrep_gold_rate:.4f})")
    print(f"Vanilla-Gold: {vanilla_correct:,}/{vanilla_total:,} ({vanilla_gold_rate:.4f})")
    
    # Analyze agreement categories for NON-GOLD tokens
    print("\nAnalyzing agreement categories (NON-GOLD tokens only)...")
    categories = analyze_agreement_categories(align_relrep, align_vanilla, gold_mappings)
    print(f"Agree: {categories['agree']:,} ({categories['agree']/categories['total']*100:.2f}%)")
    print(f"Disagree: {categories['disagree']:,} ({categories['disagree']/categories['total']*100:.2f}%)")
    print(f"Total: {categories['total']:,}")
    
    # Compile metrics
    metrics = {
        'exact_agreement_rate': exact_rate,
        'exact_matches': exact_matches,
        'total_comparisons': total_exact,
        'non_gold_total': len(non_gold_ids),
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
    plot_agreement_pie_chart(categories, os.path.join(args.figures_dir, "agreement_pie_chart.png"))
    
    # Only create Venn diagram if matplotlib_venn is available
    if venn2 is not None:
        plot_venn_diagram(metrics, os.path.join(args.figures_dir, "venn_diagram.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

