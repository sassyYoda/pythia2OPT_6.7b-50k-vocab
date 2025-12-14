"""
Word Type Analysis

Analyzes alignment agreement by word type characteristics:
- Subword vs full word
- Short vs long tokens
- High vs low frequency (if available)
"""

import json
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def load_alignment_matrix(path):
    """Load alignment matrix from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def categorize_token(token_str, token_id, tokenizer):
    """Categorize a token by its characteristics."""
    categories = []
    
    # Check if subword
    if token_str.startswith('##') or (token_str.startswith('Ġ') and len(token_str) > 1):
        categories.append('subword')
    else:
        categories.append('full_word')
    
    # Length-based categories
    clean_token = token_str.replace('Ġ', '').replace('##', '').strip()
    if len(clean_token) < 3:
        categories.append('short')
    elif len(clean_token) > 10:
        categories.append('long')
    else:
        categories.append('medium')
    
    # Special token check
    if token_str in ['<unk>', '<pad>', '<s>', '</s>', '<|endoftext|>']:
        categories.append('special')
    else:
        categories.append('regular')
    
    return categories


def compute_agreement_by_category(align_relrep, align_vanilla, target_tokenizer):
    """Compute agreement rates by token category."""
    category_tokens = defaultdict(list)
    category_agreements = defaultdict(int)
    category_totals = defaultdict(int)
    
    # Get all target IDs
    target_ids = set(align_relrep.keys()) | set(align_vanilla.keys())
    
    for target_id in target_ids:
        if target_id not in align_relrep or target_id not in align_vanilla:
            continue
        
        try:
            token_str = target_tokenizer.convert_ids_to_tokens([int(target_id)])[0]
            categories = categorize_token(token_str, target_id, target_tokenizer)
            
            # Check agreement
            agrees = (align_relrep[target_id] == align_vanilla[target_id])
            
            for category in categories:
                category_tokens[category].append(target_id)
                if agrees:
                    category_agreements[category] += 1
                category_totals[category] += 1
        except Exception as e:
            # Skip tokens that can't be decoded
            continue
    
    # Compute agreement rates
    category_rates = {}
    for category in category_totals:
        if category_totals[category] > 0:
            category_rates[category] = category_agreements[category] / category_totals[category]
        else:
            category_rates[category] = 0.0
    
    return category_rates, category_totals


def plot_word_type_analysis(category_rates, category_totals, output_path):
    """Create grouped bar chart by word type."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Organize categories
    main_categories = ['subword', 'full_word', 'short', 'medium', 'long', 'special', 'regular']
    rates = [category_rates.get(cat, 0.0) for cat in main_categories]
    totals = [category_totals.get(cat, 0) for cat in main_categories]
    
    # Filter out categories with no data
    filtered_data = [(cat, rate, total) for cat, rate, total in zip(main_categories, rates, totals) if total > 0]
    categories, rates, totals = zip(*filtered_data) if filtered_data else ([], [], [])
    
    if not categories:
        print("Warning: No valid categories found for plotting")
        return
    
    bars = ax.bar(categories, rates, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels and counts
    for bar, rate, total in zip(bars, rates, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.3f}\n(n={total})',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Agreement Rate', fontsize=12)
    ax.set_xlabel('Token Category', fontsize=12)
    ax.set_title('Alignment Agreement by Word Type', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(rates) * 1.2 if rates else 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved word type analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment by word type")
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
        "--target-tokenizer",
        type=str,
        default="facebook/opt-6.7b",
        help="Target tokenizer path"
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
    print("Word Type Analysis")
    print("="*80)
    
    # Load data
    print("\nLoading alignment matrices...")
    align_relrep = load_alignment_matrix(args.relrep_matrix)
    align_vanilla = load_alignment_matrix(args.vanilla_matrix)
    
    print("\nLoading tokenizer...")
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer, trust_remote_code=True)
    
    # Compute agreement by category
    print("\nComputing agreement by word type...")
    category_rates, category_totals = compute_agreement_by_category(
        align_relrep, align_vanilla, target_tokenizer
    )
    
    print("\nAgreement rates by category:")
    for category in sorted(category_rates.keys()):
        rate = category_rates[category]
        total = category_totals[category]
        print(f"  {category:15s}: {rate:.4f} (n={total})")
    
    # Save metrics
    metrics = {
        'category_agreement_rates': category_rates,
        'category_totals': category_totals
    }
    
    output_path = os.path.join(args.output_dir, "word_type_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_word_type_analysis(category_rates, category_totals, 
                            os.path.join(args.figures_dir, "word_type_analysis.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

