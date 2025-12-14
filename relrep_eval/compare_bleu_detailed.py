"""
BLEU Score Comparison

Compares BLEU scores between relrep and vanilla alignments.
"""

import json
import os
import argparse
import subprocess
import re
import matplotlib.pyplot as plt


def run_bleu_evaluation(matrix_path, eval_file, tokenizer_path):
    """Run BLEU evaluation using eval_matrix.py."""
    cmd = [
        'python', 'src/eval_matrix.py',
        '-e', 'bleu',
        '-m', matrix_path,
        '-f', eval_file,
        '-t', tokenizer_path,
        '-w', '1,0,0,0'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse BLEU scores from output
        bleu_scores = {}
        for line in output.split('\n'):
            if 'BLEU-1:' in line:
                match = re.search(r'BLEU-1:\s+([\d.]+)', line)
                if match:
                    bleu_scores['BLEU-1'] = float(match.group(1))
            elif 'BLEU-2:' in line:
                match = re.search(r'BLEU-2:\s+([\d.]+)', line)
                if match:
                    bleu_scores['BLEU-2'] = float(match.group(1))
            elif 'BLEU-3:' in line:
                match = re.search(r'BLEU-3:\s+([\d.]+)', line)
                if match:
                    bleu_scores['BLEU-3'] = float(match.group(1))
            elif 'BLEU-4:' in line:
                match = re.search(r'BLEU-4:\s+([\d.]+)', line)
                if match:
                    bleu_scores['BLEU-4'] = float(match.group(1))
        
        return bleu_scores
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return {}


def plot_bleu_comparison(relrep_bleus, vanilla_bleus, output_path):
    """Create grouped bar chart comparing BLEU scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    x = np.arange(len(metrics))
    width = 0.35
    
    relrep_values = [relrep_bleus.get(m, 0.0) for m in metrics]
    vanilla_values = [vanilla_bleus.get(m, 0.0) for m in metrics]
    
    bars1 = ax.bar(x - width/2, relrep_values, width, label='RelRep', 
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, vanilla_values, width, label='Vanilla', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_xlabel('BLEU Metric', fontsize=12)
    ax.set_title('BLEU Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved BLEU comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare BLEU scores")
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
        "--eval-file",
        type=str,
        default="./data/pretrain-dataset/pythia-2-opt-6.7b-glove-eval-mix",
        help="Path to evaluation data file"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Tokenizer path"
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
    print("BLEU Score Comparison")
    print("="*80)
    
    # Evaluate RelRep alignment
    print("\nEvaluating RelRep alignment...")
    relrep_bleus = run_bleu_evaluation(args.relrep_matrix, args.eval_file, args.tokenizer_path)
    
    if not relrep_bleus:
        print("Error: Could not get RelRep BLEU scores")
        return
    
    print("RelRep BLEU scores:")
    for metric, score in relrep_bleus.items():
        print(f"  {metric}: {score:.6f}")
    
    # Evaluate Vanilla alignment
    print("\nEvaluating Vanilla alignment...")
    vanilla_bleus = run_bleu_evaluation(args.vanilla_matrix, args.eval_file, args.tokenizer_path)
    
    if not vanilla_bleus:
        print("Error: Could not get Vanilla BLEU scores")
        return
    
    print("Vanilla BLEU scores:")
    for metric, score in vanilla_bleus.items():
        print(f"  {metric}: {score:.6f}")
    
    # Compare
    print("\nComparison:")
    improvements = {}
    for metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        relrep_score = relrep_bleus.get(metric, 0.0)
        vanilla_score = vanilla_bleus.get(metric, 0.0)
        improvement = relrep_score - vanilla_score
        improvements[metric] = improvement
        print(f"  {metric}: RelRep={relrep_score:.6f}, Vanilla={vanilla_score:.6f}, "
              f"Improvement={improvement:+.6f}")
    
    # Save metrics
    metrics = {
        'relrep_bleus': relrep_bleus,
        'vanilla_bleus': vanilla_bleus,
        'improvements': improvements
    }
    
    output_path = os.path.join(args.output_dir, "bleu_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_bleu_comparison(relrep_bleus, vanilla_bleus,
                        os.path.join(args.figures_dir, "bleu_comparison.png"))
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

