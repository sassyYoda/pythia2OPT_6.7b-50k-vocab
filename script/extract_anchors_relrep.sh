#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Paths matching token_align_rel.sh
export GLOVE_VECTOR_PATH1="${MAIN_DIR}/data/vec-mix-pythia.txt"
export GLOVE_VECTOR_PATH2="${MAIN_DIR}/data/vec-mix-opt-6.7b.txt"
export TGT_ID_2_SRC_ID_GOLD_PATH="${MAIN_DIR}/data/Vocab_count/opt-6.7b2pythia.json"

# Output path for anchor tokens
export ANCHOR_OUTPUT_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/align_matrix_relrep_anchors.json"

# Number of anchor tokens (should match token_align_rel.sh)
export NUM_ANCHORS=300

# Random seed (should match token_align_rel.sh)
export SEED=42

# Tokenizer paths
export TOKENIZER_PATH1="EleutherAI/pythia-1b"
export TOKENIZER_PATH2="facebook/opt-6.7b"

printf "\n### Extracting/Recreating Anchor Tokens ###\n\n"
printf "Source GloVe: ${GLOVE_VECTOR_PATH1}\n"
printf "Target GloVe: ${GLOVE_VECTOR_PATH2}\n"
printf "Gold mappings: ${TGT_ID_2_SRC_ID_GOLD_PATH}\n"
printf "Number of anchors: ${NUM_ANCHORS}\n"
printf "Random seed: ${SEED}\n"
printf "Output: ${ANCHOR_OUTPUT_PATH}\n\n"

python src/extract_anchors_relrep.py \
    -s ${GLOVE_VECTOR_PATH1} \
    -t ${GLOVE_VECTOR_PATH2} \
    -g ${TGT_ID_2_SRC_ID_GOLD_PATH} \
    -o ${ANCHOR_OUTPUT_PATH} \
    --num-anchors ${NUM_ANCHORS} \
    --seed ${SEED} \
    --source-tokenizer ${TOKENIZER_PATH1} \
    --target-tokenizer ${TOKENIZER_PATH2}

