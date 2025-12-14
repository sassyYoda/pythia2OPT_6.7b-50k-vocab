#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# git clone https://github.com/stanfordnlp/GloVe.git
export GLOVE_DIR="${GLOVE_DIR:-$(cd "${MAIN_DIR}/.." && pwd)/GloVe}"

export MODLE_PATH1="EleutherAI/pythia-1b"
export TOKENIZER_PATH1="EleutherAI/pythia-1b"
export GLOVE_TRAIN_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-pythia-glove"
export GLOVE_VECTOR_PATH1="${MAIN_DIR}/data/vec-mix-pythia.txt"

export MODLE_PATH2="facebook/opt-6.7b"
export TOKENIZER_PATH2="facebook/opt-6.7b"
export GLOVE_TRAIN_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-opt-6.7b-glove"
export GLOVE_VECTOR_PATH2="${MAIN_DIR}/data/vec-mix-opt-6.7b.txt"

export TGT_ID_2_SRC_ID_GOLD_PATH="${MAIN_DIR}/data/Vocab_count/opt-6.7b2pythia.json"
# The output path of token alignment matrix (with relative representations)
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/align_matrix_relrep.json"

# Number of anchor tokens
export NUM_ANCHORS=300

# Random seed for anchor selection
export SEED=42


# Stage-1: train glove vectors (same as original)
cd ${GLOVE_DIR}
GLOVE_VECTOR_NAME1=$(basename ${GLOVE_VECTOR_PATH1})
GLOVE_VECTOR_NAME1="${GLOVE_VECTOR_NAME1%.*}"
printf "\n### Train GloVe vector ${GLOVE_VECTOR_NAME1} with ${GLOVE_TRAIN_PATH1}  ###\n\n"
bash ${MAIN_DIR}/script/train_glove.sh ${GLOVE_TRAIN_PATH1} ${GLOVE_VECTOR_NAME1}
mv ${GLOVE_VECTOR_NAME1}.txt ${GLOVE_VECTOR_PATH1}

GLOVE_VECTOR_NAME2=$(basename ${GLOVE_VECTOR_PATH2})
GLOVE_VECTOR_NAME2="${GLOVE_VECTOR_NAME2%.*}"
printf "\n### Train GloVe vector ${GLOVE_VECTOR_NAME2} with ${GLOVE_TRAIN_PATH2}  ###\n\n"
bash ${MAIN_DIR}/script/train_glove.sh ${GLOVE_TRAIN_PATH2} ${GLOVE_VECTOR_NAME2}
mv ${GLOVE_VECTOR_NAME2}.txt ${GLOVE_VECTOR_PATH2}


# Stage-2: token ID align with relative representations
cd ${MAIN_DIR}

export VOCAB_SIZE1=$(python src/count_vocab.py -m ${MODLE_PATH1})
export VOCAB_SIZE2=$(python src/count_vocab.py -m ${MODLE_PATH2})

# Generate gold mappings (same as original)
python src/count_dict.py \
    -s ${TOKENIZER_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${TGT_ID_2_SRC_ID_GOLD_PATH}

# NEW: Use relative representations for alignment
printf "\n### Computing alignment matrix with relative representations ###\n\n"
python src/cal_trans_matrix_relrep.py \
    -s ${GLOVE_VECTOR_PATH1} \
    -s1 ${VOCAB_SIZE1} \
    -t ${GLOVE_VECTOR_PATH2} \
    -s2 ${VOCAB_SIZE2} \
    -g ${TGT_ID_2_SRC_ID_GOLD_PATH} \
    -o ${TGT_ID_2_SRC_ID_RES_PATH} \
    --num-anchors ${NUM_ANCHORS} \
    --seed ${SEED} \
    --source-tokenizer ${TOKENIZER_PATH1} \
    --target-tokenizer ${TOKENIZER_PATH2}

