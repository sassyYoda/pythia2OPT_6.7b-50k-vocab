"""
Semantic Similarity Preservation Analysis

Analyzes how well semantic similarity is preserved after alignment.
Uses WordNet/ConceptNet for semantic similarity pairs.
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
try:
    from nltk.corpus import wordnet as wn
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False
    print("Warning: NLTK WordNet not available. Using default similar pairs.")
from scipy.spatial.distance import cosine


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


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def get_similar_pairs_from_wordnet(num_pairs=100):
    """Generate similar word pairs using WordNet."""
    if not HAS_WORDNET:
        return []
    
    similar_pairs = []
    
    # Get synsets and find similar words
    synsets = list(wn.all_synsets())
    np.random.seed(42)
    
    for _ in range(num_pairs * 2):  # Generate extra to account for filtering
        if len(similar_pairs) >= num_pairs:
            break
            
        synset = np.random.choice(synsets)
        lemmas = synset.lemmas()
        
        if len(lemmas) < 2:
            continue
        
        # Get two different lemmas from same synset (synonyms)
        lemma1, lemma2 = np.random.choice(lemmas, size=2, replace=False)
        word1 = lemma1.name().replace('_', ' ')
        word2 = lemma2.name().replace('_', ' ')
        
        if word1 != word2 and (word1, word2) not in similar_pairs:
            similar_pairs.append((word1, word2))
    
    return similar_pairs[:num_pairs]


def create_default_similar_pairs():
    """Create a default set of similar word pairs."""
    return [
        ('scientist', 'researcher'),
        ('happy', 'joyful'),
        ('car', 'automobile'),
        ('big', 'large'),
        ('run', 'sprint'),
        ('house', 'home'),
        ('dog', 'puppy'),
        ('fast', 'quick'),
        ('beautiful', 'pretty'),
        ('smart', 'intelligent'),
        ('small', 'tiny'),
        ('angry', 'mad'),
        ('cold', 'freezing'),
        ('hot', 'warm'),
        ('sad', 'unhappy'),
        ('good', 'great'),
        ('bad', 'terrible'),
        ('walk', 'stroll'),
        ('talk', 'speak'),
        ('see', 'look'),
    ]


def word_to_token_id(word, tokenizer):
    """Convert word to token ID, handling multiple tokens."""
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if len(tokens) == 0:
        return None
    # Return first token ID
    return tokens[0]


def analyze_semantic_preservation(align_relrep, align_vanilla, source_glove, target_glove,
                                  similar_pairs, source_tokenizer, target_tokenizer):
    """Analyze semantic similarity preservation."""
    results = []
    
    for word_a, word_b in similar_pairs:
        # Convert to token IDs
        target_id_a = word_to_token_id(word_a, target_tokenizer)
        target_id_b = word_to_token_id(word_b, target_tokenizer)
        
        if target_id_a is None or target_id_b is None:
            continue
        
        target_id_a_str = str(target_id_a)
        target_id_b_str = str(target_id_b)
        
        # Skip if tokens not in alignment matrices
        if (target_id_a_str not in align_relrep or target_id_b_str not in align_relrep or
            target_id_a_str not in align_vanilla or target_id_b_str not in align_vanilla):
            continue
        
        # Skip if tokens not in GloVe models
        if (target_id_a_str not in target_glove or target_id_b_str not in target_glove):
            continue
        
        # Original similarity in target space
        original_sim = cosine_similarity(
            target_glove[target_id_a_str],
            target_glove[target_id_b_str]
        )
        
        # RelRep aligned similarity
        source_id_a_relrep = str(align_relrep[target_id_a_str])
        source_id_b_relrep = str(align_relrep[target_id_b_str])
        
        if source_id_a_relrep in source_glove and source_id_b_relrep in source_glove:
            relrep_sim = cosine_similarity(
                source_glove[source_id_a_relrep],
                source_glove[source_id_b_relrep]
            )
        else:
            continue
        
        # Vanilla aligned similarity
        source_id_a_vanilla = str(align_vanilla[target_id_a_str])
        source_id_b_vanilla = str(align_vanilla[target_id_b_str])
        
        if source_id_a_vanilla in source_glove and source_id_b_vanilla in source_glove:
            vanilla_sim = cosine_similarity(
                source_glove[source_id_a_vanilla],
                source_glove[source_id_b_vanilla]
            )
        else:
            continue
        
        # Compute preservation (lower is better - how much similarity changed)
        relrep_preservation = abs(original_sim - relrep_sim)
        vanilla_preservation = abs(original_sim - vanilla_sim)
        
        results.append({
            'pair': (word_a, word_b),
            'original_sim': original_sim,
            'relrep_sim': relrep_sim,
            'vanilla_sim': vanilla_sim,
            'relrep_preservation': relrep_preservation,
            'vanilla_preservation': vanilla_preservation
        })
    
    return results


def plot_semantic_preservation(results, output_path):
    """Create scatter plots showing semantic preservation."""
    if not results:
        print("Warning: No results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    original_sims = [r['original_sim'] for r in results]
    relrep_sims = [r['relrep_sim'] for r in results]
    vanilla_sims = [r['vanilla_sim'] for r in results]
    
    # Plot 1: RelRep preservation
    ax1 = axes[0]
    ax1.scatter(original_sims, relrep_sims, alpha=0.6, color='#2ecc71', s=50)
    
    # Diagonal line (perfect preservation)
    min_val = min(min(original_sims), min(relrep_sims))
    max_val = max(max(original_sims), max(relrep_sims))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Preservation')
    
    ax1.set_xlabel('Original Similarity (Target Space)', fontsize=11)
    ax1.set_ylabel('Aligned Similarity (Source Space)', fontsize=11)
    ax1.set_title('RelRep Semantic Preservation', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot 2: Vanilla preservation
    ax2 = axes[1]
    ax2.scatter(original_sims, vanilla_sims, alpha=0.6, color='#e74c3c', s=50)
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Preservation')
    
    ax2.set_xlabel('Original Similarity (Target Space)', fontsize=11)
    ax2.set_ylabel('Aligned Similarity (Source Space)', fontsize=11)
    ax2.set_title('Vanilla Semantic Preservation', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved semantic preservation plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic similarity preservation")
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
        "--similar-pairs-file",
        type=str,
        default=None,
        help="Path to file with similar pairs (format: word1,word2 per line). If not provided, uses default pairs."
    )
    parser.add_argument(
        "--source-tokenizer",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Source tokenizer path"
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
    print("Semantic Similarity Preservation Analysis")
    print("="*80)
    
    # Load similar pairs
    if args.similar_pairs_file and os.path.exists(args.similar_pairs_file):
        print(f"\nLoading similar pairs from {args.similar_pairs_file}...")
        similar_pairs = []
        with open(args.similar_pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    similar_pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        print("\nUsing default similar pairs...")
        similar_pairs = create_default_similar_pairs()
    
    print(f"Loaded {len(similar_pairs)} similar pairs")
    
    # Load data
    print("\nLoading alignment matrices...")
    align_relrep = load_alignment_matrix(args.relrep_matrix)
    align_vanilla = load_alignment_matrix(args.vanilla_matrix)
    
    print("\nLoading GloVe embeddings...")
    source_glove = load_glove_model(args.source_glove)
    target_glove = load_glove_model(args.target_glove)
    
    print("\nLoading tokenizers...")
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer, trust_remote_code=True)
    
    # Analyze preservation
    print("\nAnalyzing semantic preservation...")
    results = analyze_semantic_preservation(
        align_relrep, align_vanilla, source_glove, target_glove,
        similar_pairs, source_tokenizer, target_tokenizer
    )
    
    print(f"Analyzed {len(results)} pairs")
    
    if results:
        # Compute statistics
        mean_relrep_preservation = np.mean([r['relrep_preservation'] for r in results])
        mean_vanilla_preservation = np.mean([r['vanilla_preservation'] for r in results])
        
        print(f"\nMean preservation error:")
        print(f"  RelRep: {mean_relrep_preservation:.4f}")
        print(f"  Vanilla: {mean_vanilla_preservation:.4f}")
        
        # Save metrics
        metrics = {
            'num_pairs_analyzed': len(results),
            'mean_relrep_preservation_error': float(mean_relrep_preservation),
            'mean_vanilla_preservation_error': float(mean_vanilla_preservation),
            'results': results
        }
        
        output_path = os.path.join(args.output_dir, "semantic_preservation_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {output_path}")
        
        # Create visualization
        print("\nCreating visualization...")
        plot_semantic_preservation(results, os.path.join(args.figures_dir, "semantic_preservation.png"))
    else:
        print("\nWarning: No valid pairs could be analyzed")
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

