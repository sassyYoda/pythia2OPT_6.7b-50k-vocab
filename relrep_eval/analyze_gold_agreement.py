"""
Gold Label Agreement Breakdown

Analyzes agreement with gold mappings in detail:
- Confusion matrix style analysis
- Venn diagram visualization
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
try:
    from matplotlib_venn import venn2
except ImportError:
    print("Warning: matplotlib_venn not installed. Install with: pip install matplotlib-venn")
    venn2 = None


def load_alignment_matrix(path):
    """Load alignment matrix from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_gold_agreement(align_relrep, align_vanilla, gold_mappings):
    """Analyze agreement with gold mappings."""
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
            text = ax.text(j, i, f'{matrix[i, j]:,}',
                          ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Vanilla\nCorrect', 'Vanilla\nWrong'], fontsize=12)
    ax.set_yticklabels(['RelRep\nCorrect', 'RelRep\nWrong'], fontsize=12)
    
    ax.set_xlabel('Vanilla Alignment', fontsize=13, fontweight='bold')
    ax.set_ylabel('RelRep Alignment', fontsize=13, fontweight='bold')
    ax.set_title('Agreement with Gold Mappings\n(Confusion Matrix)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_venn_diagram(categories, total_gold, output_path):
    """Create Venn diagram showing agreement overlap."""
    if venn2 is None:
        print("Warning: Skipping Venn diagram (matplotlib_venn not available)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate sets
    relrep_correct = categories['both_correct'] + categories['only_relrep_correct']
    vanilla_correct = categories['both_correct'] + categories['only_vanilla_correct']
    both_correct = categories['both_correct']
    
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
    
    ax.set_title('Agreement with Gold Mappings\n(Venn Diagram)', fontsize=14, fontweight='bold')
    
    # Add summary text
    summary_text = (
        f"Total Gold Mappings: {total_gold:,}\n"
        f"Both Correct: {both_correct:,}\n"
        f"Only RelRep: {categories['only_relrep_correct']:,}\n"
        f"Only Vanilla: {categories['only_vanilla_correct']:,}\n"
        f"Both Wrong: {categories['both_wrong']:,}"
    )
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Venn diagram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze gold agreement breakdown")
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
    print("Gold Label Agreement Breakdown")
    print("="*80)
    
    # Load data
    print("\nLoading alignment matrices...")
    align_relrep = load_alignment_matrix(args.relrep_matrix)
    align_vanilla = load_alignment_matrix(args.vanilla_matrix)
    
    print("\nLoading gold mappings...")
    with open(args.gold_mappings, 'r') as f:
        gold_mappings = json.load(f)
    print(f"Gold mappings: {len(gold_mappings)} pairs")
    
    # Analyze agreement
    print("\nAnalyzing gold agreement...")
    categories = analyze_gold_agreement(align_relrep, align_vanilla, gold_mappings)
    
    total = sum(categories.values())
    print(f"\nAgreement Categories:")
    print(f"  Both correct: {categories['both_correct']:,} ({categories['both_correct']/total*100:.2f}%)")
    print(f"  Only RelRep correct: {categories['only_relrep_correct']:,} ({categories['only_relrep_correct']/total*100:.2f}%)")
    print(f"  Only Vanilla correct: {categories['only_vanilla_correct']:,} ({categories['only_vanilla_correct']/total*100:.2f}%)")
    print(f"  Both wrong: {categories['both_wrong']:,} ({categories['both_wrong']/total*100:.2f}%)")
    print(f"    - But agree: {categories['both_wrong_but_agree']:,}")
    print(f"    - And disagree: {categories['both_wrong_and_disagree']:,}")
    
    # Compute rates
    relrep_rate = (categories['both_correct'] + categories['only_relrep_correct']) / total
    vanilla_rate = (categories['both_correct'] + categories['only_vanilla_correct']) / total
    
    print(f"\nOverall Accuracy:")
    print(f"  RelRep: {relrep_rate:.4f}")
    print(f"  Vanilla: {vanilla_rate:.4f}")
    
    # Save metrics
    metrics = {
        'categories': categories,
        'total_gold_mappings': len(gold_mappings),
        'relrep_accuracy': float(relrep_rate),
        'vanilla_accuracy': float(vanilla_rate)
    }
    
    output_path = os.path.join(args.output_dir, "gold_agreement_stats.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_confusion_matrix(categories, os.path.join(args.figures_dir, "gold_agreement_matrix.png"))
    plot_venn_diagram(categories, len(gold_mappings), 
                    os.path.join(args.figures_dir, "gold_agreement_venn.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

