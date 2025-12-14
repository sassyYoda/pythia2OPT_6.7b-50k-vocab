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


def get_learned_tokens(align_relrep, align_vanilla, gold_mappings, vocab_size=50272):
    """Get tokens that are NOT in gold mappings (i.e., aligned by methodology).
    
    CRITICAL: We need to use the full vocabulary range, not just what's in the alignment matrices.
    """
    # Get all target token IDs from full vocabulary
    all_target_tokens = set(range(vocab_size))
    
    # Get gold-mapped token IDs (convert keys to int for comparison)
    gold_tokens = set(int(k) for k in gold_mappings.keys())
    
    # Learned tokens = all tokens - gold tokens
    learned_tokens = all_target_tokens - gold_tokens
    
    return learned_tokens, gold_tokens, all_target_tokens


def compute_exact_agreement(align_relrep, align_vanilla, learned_tokens):
    """Compute exact agreement rate between two alignment matrices for LEARNED tokens only."""
    agreement_count = 0
    total = 0
    
    for token_id in learned_tokens:
        token_id_str = str(token_id)
        
        if token_id_str in align_relrep and token_id_str in align_vanilla:
            relrep_choice = align_relrep[token_id_str]
            vanilla_choice = align_vanilla[token_id_str]
            
            if relrep_choice == vanilla_choice:
                agreement_count += 1
            total += 1
    
    exact_agreement_rate = agreement_count / total if total > 0 else 0.0
    return agreement_count, total, exact_agreement_rate


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


def analyze_learned_tokens(align_relrep, align_vanilla, learned_tokens, gold_mappings):
    """Comprehensive analysis of learned tokens."""
    agreements = []
    disagreements = []
    
    # Get gold source tokens
    gold_source_tokens = set(gold_mappings.values())
    
    relrep_uses_gold_sources = 0
    vanilla_uses_gold_sources = 0
    
    relrep_sources = []
    vanilla_sources = []
    
    for token_id in learned_tokens:
        token_id_str = str(token_id)
        
        if token_id_str not in align_relrep or token_id_str not in align_vanilla:
            continue
        
        relrep_choice = align_relrep[token_id_str]
        vanilla_choice = align_vanilla[token_id_str]
        
        relrep_sources.append(relrep_choice)
        vanilla_sources.append(vanilla_choice)
        
        if relrep_choice == vanilla_choice:
            agreements.append(token_id)
        else:
            disagreements.append(token_id)
        
        # Check if mapping uses gold source tokens
        if relrep_choice in gold_source_tokens:
            relrep_uses_gold_sources += 1
        if vanilla_choice in gold_source_tokens:
            vanilla_uses_gold_sources += 1
    
    total_analyzed = len(agreements) + len(disagreements)
    
    # Compute diversity (unique sources)
    relrep_unique_sources = len(set(relrep_sources))
    vanilla_unique_sources = len(set(vanilla_sources))
    
    results = {
        'agreements': agreements,
        'disagreements': disagreements,
        'total_analyzed': total_analyzed,
        'relrep_uses_gold_sources': relrep_uses_gold_sources,
        'vanilla_uses_gold_sources': vanilla_uses_gold_sources,
        'relrep_gold_source_rate': relrep_uses_gold_sources / total_analyzed if total_analyzed > 0 else 0.0,
        'vanilla_gold_source_rate': vanilla_uses_gold_sources / total_analyzed if total_analyzed > 0 else 0.0,
        'relrep_unique_sources': relrep_unique_sources,
        'vanilla_unique_sources': vanilla_unique_sources,
        'relrep_diversity': relrep_unique_sources / total_analyzed if total_analyzed > 0 else 0.0,
        'vanilla_diversity': vanilla_unique_sources / total_analyzed if total_analyzed > 0 else 0.0
    }
    
    return results


def plot_agreement_learned_only(metrics, learned_analysis, output_path):
    """Create visualization focusing on learned tokens only."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Simple bar chart
    exact_agreement_rate = metrics['exact_agreement_rate']
    categories = ['Exact Agreement\n(Learned Tokens)']
    
    bars = ax1.bar(categories, [exact_agreement_rate], color='steelblue', width=0.4, alpha=0.7, edgecolor='black')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Agreement Rate', fontsize=12)
    ax1.set_title('Agreement on Learned Tokens Only', fontsize=13, fontweight='bold')
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random baseline', linewidth=2)
    ax1.text(0, exact_agreement_rate + 0.03, f'{exact_agreement_rate:.4f}', 
             ha='center', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right: Breakdown pie chart
    labels = ['Both Agree', 'Methods Disagree']
    sizes = [len(learned_analysis['agreements']), len(learned_analysis['disagreements'])]
    colors = ['#66c2a5', '#fc8d62']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Agreement Distribution on Learned Tokens', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved agreement learned-only plot to {output_path}")


def plot_additional_metrics(learned_analysis, output_path):
    """Plot additional quality metrics for learned tokens."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Overlap with gold source space
    ax1 = axes[0]
    categories = ['RelRep', 'Vanilla']
    values = [
        learned_analysis['relrep_gold_source_rate'],
        learned_analysis['vanilla_gold_source_rate']
    ]
    bars = ax1.bar(categories, values, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Rate', fontsize=11)
    ax1.set_title('Overlap with Gold Source Space', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Alignment diversity
    ax2 = axes[1]
    values = [
        learned_analysis['relrep_diversity'],
        learned_analysis['vanilla_diversity']
    ]
    bars = ax2.bar(categories, values, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Diversity (Unique Sources / Total)', fontsize=11)
    ax2.set_title('Alignment Diversity', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Summary statistics text
    ax3 = axes[2]
    ax3.axis('off')
    stats_text = (
        f"Learned Token Analysis Summary\n"
        f"{'='*40}\n\n"
        f"Total Learned Tokens: {learned_analysis['total_analyzed']:,}\n\n"
        f"Agreements: {len(learned_analysis['agreements']):,}\n"
        f"Disagreements: {len(learned_analysis['disagreements']):,}\n\n"
        f"RelRep Unique Sources: {learned_analysis['relrep_unique_sources']:,}\n"
        f"Vanilla Unique Sources: {learned_analysis['vanilla_unique_sources']:,}\n\n"
        f"RelRep → Gold Sources: {learned_analysis['relrep_uses_gold_sources']:,}\n"
        f"Vanilla → Gold Sources: {learned_analysis['vanilla_uses_gold_sources']:,}"
    )
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved additional metrics plot to {output_path}")


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
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50272,
        help="Full target vocabulary size"
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
    
    # CRITICAL: Identify learned tokens using full vocabulary
    print("\nIdentifying learned tokens...")
    learned_tokens, gold_tokens, all_target_tokens = get_learned_tokens(
        align_relrep, align_vanilla, gold_mappings, args.vocab_size
    )
    
    print(f"Total tokens: {len(all_target_tokens):,}")
    print(f"Gold-mapped tokens: {len(gold_tokens):,}")
    print(f"Learned tokens (aligned by methodology): {len(learned_tokens):,}")
    
    # Compute exact agreement for LEARNED tokens only
    print("\n=== Exact Agreement (Learned Tokens Only) ===")
    agreement_count, total_analyzed, exact_agreement_rate = compute_exact_agreement(
        align_relrep, align_vanilla, learned_tokens
    )
    print(f"Agreement: {agreement_count:,}/{total_analyzed:,} = {exact_agreement_rate:.4f}")
    
    # Verify gold mappings are preserved (should be 100% or close)
    print("\n=== Verifying Gold Mappings are Preserved ===")
    relrep_correct, relrep_total, relrep_gold_rate = compute_gold_agreement(align_relrep, gold_mappings)
    vanilla_correct, vanilla_total, vanilla_gold_rate = compute_gold_agreement(align_vanilla, gold_mappings)
    
    print(f"RelRep-Gold: {relrep_correct:,}/{relrep_total:,} ({relrep_gold_rate:.4f})")
    print(f"Vanilla-Gold: {vanilla_correct:,}/{vanilla_total:,} ({vanilla_gold_rate:.4f})")
    
    # Comprehensive analysis of learned tokens
    print("\n=== Comprehensive Learned Token Analysis ===")
    learned_analysis = analyze_learned_tokens(align_relrep, align_vanilla, learned_tokens, gold_mappings)
    
    print(f"\nAgreement/Disagreement Split:")
    print(f"  Agreements: {len(learned_analysis['agreements']):,} ({len(learned_analysis['agreements'])/learned_analysis['total_analyzed']*100:.2f}%)")
    print(f"  Disagreements: {len(learned_analysis['disagreements']):,} ({len(learned_analysis['disagreements'])/learned_analysis['total_analyzed']*100:.2f}%)")
    
    print(f"\nOverlap with Gold Source Space:")
    print(f"  RelRep maps to gold sources: {learned_analysis['relrep_uses_gold_sources']:,}/{learned_analysis['total_analyzed']:,} = {learned_analysis['relrep_gold_source_rate']:.4f}")
    print(f"  Vanilla maps to gold sources: {learned_analysis['vanilla_uses_gold_sources']:,}/{learned_analysis['total_analyzed']:,} = {learned_analysis['vanilla_gold_source_rate']:.4f}")
    
    print(f"\nAlignment Diversity:")
    print(f"  RelRep unique sources: {learned_analysis['relrep_unique_sources']:,}/{learned_analysis['total_analyzed']:,} = {learned_analysis['relrep_diversity']:.4f}")
    print(f"  Vanilla unique sources: {learned_analysis['vanilla_unique_sources']:,}/{learned_analysis['total_analyzed']:,} = {learned_analysis['vanilla_diversity']:.4f}")
    
    # Compile metrics
    metrics = {
        'exact_agreement_rate': exact_agreement_rate,
        'agreement_count': agreement_count,
        'total_learned_tokens': total_analyzed,
        'learned_tokens_total': len(learned_tokens),
        'relrep_gold_agreement_rate': relrep_gold_rate,
        'relrep_gold_correct': relrep_correct,
        'relrep_gold_total': relrep_total,
        'vanilla_gold_agreement_rate': vanilla_gold_rate,
        'vanilla_gold_correct': vanilla_correct,
        'vanilla_gold_total': vanilla_total,
        'total_gold_mappings': len(gold_mappings),
        'learned_analysis': learned_analysis
    }
    
    # Save metrics
    output_path = os.path.join(args.output_dir, "alignment_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Save agreement/disagreement lists
    agreements_path = os.path.join(args.output_dir, "learned_tokens_agree.json")
    disagreements_path = os.path.join(args.output_dir, "learned_tokens_disagree.json")
    with open(agreements_path, 'w') as f:
        json.dump([int(tid) for tid in learned_analysis['agreements']], f, indent=2)
    with open(disagreements_path, 'w') as f:
        json.dump([int(tid) for tid in learned_analysis['disagreements']], f, indent=2)
    print(f"Saved agreement/disagreement lists")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_agreement_learned_only(metrics, learned_analysis, 
                               os.path.join(args.figures_dir, "agreement_learned_only.png"))
    plot_additional_metrics(learned_analysis, 
                           os.path.join(args.figures_dir, "additional_metrics.png"))
    
    # Only create Venn diagram if matplotlib_venn is available
    if venn2 is not None:
        plot_venn_diagram(metrics, os.path.join(args.figures_dir, "venn_diagram.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

