"""
Token alignment with relative representations.

This script applies relative representations before computing the alignment matrix.
The key insight: instead of directly comparing GloVe embeddings via cosine similarity,
we first convert them to relative representations (similarities to anchor tokens),
then compute alignment based on these relative representations.

Based on: "Relative Representations Enable Zero-Shot Latent Space Communication"
https://openreview.net/pdf?id=SrC-nwieGJ
"""

import numpy as np
import json
from tqdm import tqdm
import random
import argparse
import os
from transformers import AutoTokenizer


def load_glove_model(File):
    """Load GloVe embeddings from file."""
    print(f"Loading GloVe model from {File}...")
    glove_model = {}
    with open(File, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def normalize(x):
    """Normalize vector to unit length."""
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def is_valid_anchor_token(token_id, tokenizer, vocab):
    """
    Check if a token is valid for use as an anchor.
    
    Valid anchors should be:
    - Full words (not subword pieces)
    - Not filler words (the, a, an, is, are, etc.)
    - Not punctuation
    """
    try:
        # Convert token ID to string
        token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        
        # Check if it's a subword piece (contains special markers)
        # GPT-2/OPT style: starts with 'Ġ' (space prefix)
        # BERT style: starts with '##'
        if token.startswith('##'):
            # BERT-style subword piece - not valid
            return False
        
        # For GPT-2/OPT, 'Ġ' indicates start of word, which is actually good
        # Remove special prefixes for checking
        clean_token = token.replace('Ġ', '').replace('##', '').strip()
        
        # Must have some alphabetic content
        if not any(c.isalpha() for c in clean_token):
            return False
        
        # Check if it's punctuation only
        if all(not c.isalnum() for c in clean_token):
            return False
        
        # Check if it's a filler word
        filler_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their',
            'this', 'that', 'these', 'those',
            'what', 'which', 'who', 'when', 'where', 'why', 'how',
        }
        
        if clean_token.lower() in filler_words:
            return False
        
        # Must be at least 2 characters
        if len(clean_token) < 2:
            return False
        
        return True
        
    except Exception as e:
        print(f"Error checking token {token_id}: {e}")
        return False


def select_anchor_tokens(
    gold_mappings,
    source_glove,
    target_glove,
    source_tokenizer,
    target_tokenizer,
    num_anchors=300,
    seed=42
):
    """
    Select anchor tokens that are:
    1. Present in both vocabularies (via gold mappings)
    2. Have embeddings in both GloVe models
    3. Are valid (full words, not filler, not punctuation)
    
    Returns:
        source_anchor_ids: list of token IDs in source vocab
        target_anchor_ids: list of token IDs in target vocab
    """
    print(f"\nSelecting {num_anchors} anchor tokens...")
    random.seed(seed)
    
    # Get all candidate token IDs from gold mappings
    candidate_target_ids = []
    
    for target_id_str, source_id in gold_mappings.items():
        target_id = int(target_id_str)
        
        # Check if both IDs have embeddings
        if target_id_str not in target_glove:
            continue
        if str(source_id) not in source_glove:
            continue
        
        # Check if target token is valid
        if not is_valid_anchor_token(target_id, target_tokenizer, target_glove):
            continue
        
        # Check if source token is valid
        if not is_valid_anchor_token(source_id, source_tokenizer, source_glove):
            continue
        
        candidate_target_ids.append(target_id)
    
    print(f"Found {len(candidate_target_ids)} valid candidate anchors")
    
    if len(candidate_target_ids) < num_anchors:
        print(f"Warning: Only found {len(candidate_target_ids)} valid anchors, less than requested {num_anchors}")
        num_anchors = len(candidate_target_ids)
    
    # Randomly select anchors
    random.shuffle(candidate_target_ids)
    selected_target_ids = candidate_target_ids[:num_anchors]
    
    # Get corresponding source IDs
    selected_source_ids = [gold_mappings[str(tid)] for tid in selected_target_ids]
    
    print(f"Selected {len(selected_target_ids)} anchor tokens")
    
    # Print some example anchors
    print("\nExample anchor tokens:")
    for i in range(min(10, len(selected_target_ids))):
        target_token = target_tokenizer.convert_ids_to_tokens([selected_target_ids[i]])[0]
        source_token = source_tokenizer.convert_ids_to_tokens([selected_source_ids[i]])[0]
        print(f"  {i+1}. Target: {target_token} <-> Source: {source_token}")
    
    return selected_source_ids, selected_target_ids


def compute_relative_representation(embedding, anchor_embeddings):
    """
    Compute relative representation of an embedding with respect to anchors.
    
    Args:
        embedding: (D,) embedding vector
        anchor_embeddings: (K, D) anchor embedding matrix
    
    Returns:
        rel_rep: (K,) relative representation (cosine similarities to anchors)
    """
    # Normalize embedding
    embedding_norm = normalize(embedding)
    
    # Normalize anchors
    anchor_norms = np.array([normalize(anchor) for anchor in anchor_embeddings])
    
    # Compute cosine similarities (dot product of normalized vectors)
    rel_rep = np.dot(anchor_norms, embedding_norm)
    
    return rel_rep


def convert_to_relative_representations(
    glove_model,
    anchor_ids,
    vocab_size
):
    """
    Convert all embeddings in vocabulary to relative representations.
    
    Args:
        glove_model: dict mapping token_id (str) -> embedding
        anchor_ids: list of token IDs to use as anchors
        vocab_size: total vocabulary size
    
    Returns:
        ids: list of token IDs (str)
        rel_reps: (vocab_size, num_anchors) matrix of relative representations
    """
    print(f"Converting embeddings to relative representations...")
    
    # Get anchor embeddings
    anchor_embeddings = []
    for anchor_id in anchor_ids:
        anchor_id_str = str(anchor_id)
        if anchor_id_str in glove_model:
            anchor_embeddings.append(glove_model[anchor_id_str])
        else:
            print(f"Warning: Anchor {anchor_id_str} not in GloVe model")
    
    if len(anchor_embeddings) == 0:
        raise ValueError("No valid anchor embeddings found!")
    
    anchor_embeddings = np.array(anchor_embeddings)  # (K, D)
    num_anchors = len(anchor_embeddings)
    
    print(f"Using {num_anchors} anchors with dimension {anchor_embeddings.shape[1]}")
    
    # Compute relative representations for all tokens
    ids = []
    rel_reps = []
    
    for token_id in tqdm(range(vocab_size), desc="Computing relative representations"):
        token_id_str = str(token_id)
        ids.append(token_id_str)
        
        if token_id_str in glove_model:
            embedding = glove_model[token_id_str]
            rel_rep = compute_relative_representation(embedding, anchor_embeddings)
            rel_reps.append(rel_rep)
        else:
            # Missing embedding - use zero vector
            rel_reps.append(np.zeros(num_anchors))
    
    rel_reps = np.array(rel_reps)  # (vocab_size, num_anchors)
    
    print(f"Relative representations shape: {rel_reps.shape}")
    
    return ids, rel_reps


def compute_alignment_matrix(
    source_ids,
    source_rel_reps,
    target_ids,
    target_rel_reps,
    gold_mappings,
    source_vocab_size,
    target_vocab_size
):
    """
    Compute alignment matrix using relative representations.
    
    Instead of cosine similarity between original embeddings,
    we compute cosine similarity between relative representations.
    """
    print(f"\nComputing alignment matrix...")
    
    # Normalize relative representations for cosine similarity
    source_rel_reps_norm = np.array([
        normalize(rep) if np.linalg.norm(rep) > 0 else rep 
        for rep in source_rel_reps
    ])
    
    target_rel_reps_norm = np.array([
        normalize(rep) if np.linalg.norm(rep) > 0 else rep
        for rep in target_rel_reps
    ])
    
    # Compute similarity matrix: (target_vocab, source_vocab)
    # sim[i, j] = cosine similarity between target[i] and source[j] relative reps
    print("Computing similarity matrix...")
    sim = np.matmul(target_rel_reps_norm, source_rel_reps_norm.T)
    
    print(f"Similarity matrix shape: {sim.shape}")
    
    # Create alignment dictionary
    alignment = {}
    num_gold = 0
    num_missing = 0
    num_aligned = 0
    
    for target_id in tqdm(range(target_vocab_size), desc="Creating alignment"):
        target_id_str = str(target_id)
        
        # Use gold mapping if available
        if target_id_str in gold_mappings:
            alignment[target_id_str] = gold_mappings[target_id_str]
            num_gold += 1
            continue
        
        # Check if target token has valid relative representation
        if target_id_str not in target_ids:
            # Missing token - random assignment
            alignment[target_id_str] = random.randint(0, source_vocab_size - 1)
            num_missing += 1
            continue
        
        # Find best matching source token via relative representations
        target_idx = target_ids.index(target_id_str)
        source_idx = np.argmax(sim[target_idx])
        source_id_str = source_ids[source_idx]
        
        # Avoid mapping to unknown tokens
        if source_id_str == 'unk' or source_id_str == '<unk>':
            # Get second best
            top_2_indices = np.argsort(sim[target_idx])[-2:]
            for idx in reversed(top_2_indices):
                source_id_str = source_ids[idx]
                if source_id_str != 'unk' and source_id_str != '<unk>':
                    break
        
        alignment[target_id_str] = int(source_id_str)
        num_aligned += 1
    
    print(f"\nAlignment statistics:")
    print(f"  Gold mappings: {num_gold}")
    print(f"  Aligned via relative representations: {num_aligned}")
    print(f"  Random (missing): {num_missing}")
    print(f"  Total: {len(alignment)}")
    
    return alignment


def main():
    parser = argparse.ArgumentParser(
        description="Token alignment using relative representations"
    )
    parser.add_argument(
        "-s", "--source-glove-vector-path",
        type=str,
        required=True,
        help="Path to source GloVe vectors"
    )
    parser.add_argument(
        "-s1", "--source-vocab-size",
        type=int,
        required=True,
        help="Source vocabulary size"
    )
    parser.add_argument(
        "-t", "--target-glove-vector-path",
        type=str,
        required=True,
        help="Path to target GloVe vectors"
    )
    parser.add_argument(
        "-s2", "--target-vocab-size",
        type=int,
        required=True,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "-g", "--gold-target-to-source-path",
        type=str,
        required=True,
        help="Path to gold alignment mappings (target -> source)"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=True,
        help="Output path for alignment matrix"
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=300,
        help="Number of anchor tokens to use (default: 300)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for anchor selection (default: 42)"
    )
    parser.add_argument(
        "--source-tokenizer",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Source model tokenizer path"
    )
    parser.add_argument(
        "--target-tokenizer",
        type=str,
        default="facebook/opt-6.7b",
        help="Target model tokenizer path"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Token Alignment with Relative Representations")
    print("="*80)
    print(f"Source GloVe: {args.source_glove_vector_path}")
    print(f"Target GloVe: {args.target_glove_vector_path}")
    print(f"Number of anchors: {args.num_anchors}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_tokenizer,
        trust_remote_code=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_tokenizer,
        trust_remote_code=True
    )
    
    # Load GloVe embeddings
    source_glove = load_glove_model(args.source_glove_vector_path)
    target_glove = load_glove_model(args.target_glove_vector_path)
    
    # Load gold mappings
    print(f"\nLoading gold mappings from {args.gold_target_to_source_path}...")
    with open(args.gold_target_to_source_path, 'r') as f:
        gold_mappings = json.load(f)
    print(f"Loaded {len(gold_mappings)} gold mappings")
    
    # Step 1: Select anchor tokens
    source_anchor_ids, target_anchor_ids = select_anchor_tokens(
        gold_mappings=gold_mappings,
        source_glove=source_glove,
        target_glove=target_glove,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        num_anchors=args.num_anchors,
        seed=args.seed
    )
    
    # Step 2: Convert to relative representations
    print("\n" + "="*80)
    print("Converting source embeddings to relative representations...")
    source_ids, source_rel_reps = convert_to_relative_representations(
        glove_model=source_glove,
        anchor_ids=source_anchor_ids,
        vocab_size=args.source_vocab_size
    )
    
    print("\nConverting target embeddings to relative representations...")
    target_ids, target_rel_reps = convert_to_relative_representations(
        glove_model=target_glove,
        anchor_ids=target_anchor_ids,
        vocab_size=args.target_vocab_size
    )
    
    # Step 3: Compute alignment using relative representations
    print("\n" + "="*80)
    alignment = compute_alignment_matrix(
        source_ids=source_ids,
        source_rel_reps=source_rel_reps,
        target_ids=target_ids,
        target_rel_reps=target_rel_reps,
        gold_mappings=gold_mappings,
        source_vocab_size=args.source_vocab_size,
        target_vocab_size=args.target_vocab_size
    )
    
    # Save alignment matrix
    print(f"\nSaving alignment matrix to {args.output_path}...")
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(alignment, f, indent=2)
    
    # Save anchor tokens for reference
    anchor_output_path = args.output_path.replace('.json', '_anchors.json')
    anchor_data = {
        "num_anchors": len(source_anchor_ids),
        "seed": args.seed,
        "source_anchor_ids": source_anchor_ids,
        "target_anchor_ids": target_anchor_ids,
        "source_anchor_tokens": [
            source_tokenizer.convert_ids_to_tokens([aid])[0] 
            for aid in source_anchor_ids
        ],
        "target_anchor_tokens": [
            target_tokenizer.convert_ids_to_tokens([aid])[0] 
            for aid in target_anchor_ids
        ],
        "anchor_pairs": [
            {
                "source_id": source_anchor_ids[i],
                "target_id": target_anchor_ids[i],
                "source_token": source_tokenizer.convert_ids_to_tokens([source_anchor_ids[i]])[0],
                "target_token": target_tokenizer.convert_ids_to_tokens([target_anchor_ids[i]])[0]
            }
            for i in range(len(source_anchor_ids))
        ]
    }
    
    print(f"Saving anchor tokens to {anchor_output_path}...")
    with open(anchor_output_path, 'w') as f:
        json.dump(anchor_data, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Token alignment with relative representations complete!")
    print(f"✓ Alignment matrix saved to: {args.output_path}")
    print(f"✓ Anchor tokens saved to: {anchor_output_path}")
    print("="*80)


if __name__ == '__main__':
    main()

