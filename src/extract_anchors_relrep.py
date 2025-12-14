"""
Extract/recreate anchor tokens from existing relative representation alignment run.

This script can recreate the anchor tokens that were used if you know:
- The random seed (default: 42)
- The number of anchors (default: 300)
- The same input files (gold mappings, GloVe vectors, tokenizers)
"""

import numpy as np
import json
import random
import argparse
import os
import sys
from transformers import AutoTokenizer

# Import functions from cal_trans_matrix_relrep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cal_trans_matrix_relrep import (
    load_glove_model,
    is_valid_anchor_token,
    select_anchor_tokens
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract/recreate anchor tokens from relative representation alignment"
    )
    parser.add_argument(
        "-s", "--source-glove-vector-path",
        type=str,
        required=True,
        help="Path to source GloVe vectors"
    )
    parser.add_argument(
        "-t", "--target-glove-vector-path",
        type=str,
        required=True,
        help="Path to target GloVe vectors"
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
        help="Output path for anchor tokens JSON file"
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=300,
        help="Number of anchor tokens (default: 300)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for anchor selection (default: 42)"
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
    print("Extracting/Recreating Anchor Tokens")
    print("="*80)
    print(f"Source GloVe: {args.source_glove_vector_path}")
    print(f"Target GloVe: {args.target_glove_vector_path}")
    print(f"Gold mappings: {args.gold_target_to_source_path}")
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
    
    # Select anchor tokens (same function as in cal_trans_matrix_relrep.py)
    source_anchor_ids, target_anchor_ids = select_anchor_tokens(
        gold_mappings=gold_mappings,
        source_glove=source_glove,
        target_glove=target_glove,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        num_anchors=args.num_anchors,
        seed=args.seed
    )
    
    # Create anchor data structure
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
    
    # Save anchor tokens
    print(f"\nSaving anchor tokens to {args.output_path}...")
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(anchor_data, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Anchor tokens extracted/recreated successfully!")
    print(f"✓ Saved to: {args.output_path}")
    print("="*80)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total anchors: {len(source_anchor_ids)}")
    print(f"  First 10 anchor pairs:")
    for i in range(min(10, len(anchor_data["anchor_pairs"]))):
        pair = anchor_data["anchor_pairs"][i]
        print(f"    {i+1}. Source: {pair['source_token']} (ID: {pair['source_id']}) <-> "
              f"Target: {pair['target_token']} (ID: {pair['target_id']})")


if __name__ == '__main__':
    main()

