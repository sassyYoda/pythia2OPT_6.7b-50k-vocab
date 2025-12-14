#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# The path of relative representation token alignment matrix
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/align_matrix_relrep.json"
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/align_matrix_relrep_demo.json"

export MATRIX_EVAL_DATA_PATH="${MAIN_DIR}/data/pretrain-dataset/pythia-2-opt-6.7b-glove-eval-mix"

# BLEU-1 evaluation
export EVAL_METHOD=bleu
export BLEU_WEIGHT="1,0,0,0"

# Bert-score evaluation
# export EVAL_METHOD=bert-score
export BERT_SOCRE_EVAL_MODEL="all-mpnet-base-v2"
export TOKENIZER_PATH="EleutherAI/pythia-1b"

printf "\n### Evaluating Relative Representation Alignment Matrix ###\n\n"
printf "Alignment matrix: ${TGT_ID_2_SRC_ID_RES_PATH}\n"
printf "Evaluation data: ${MATRIX_EVAL_DATA_PATH}\n"
printf "Evaluation method: ${EVAL_METHOD}\n"
if [ "${EVAL_METHOD}" = "bleu" ]; then
    printf "BLEU weights: ${BLEU_WEIGHT}\n"
else
    printf "BERT-score model: ${BERT_SOCRE_EVAL_MODEL}\n"
fi
printf "\n"

python src/eval_matrix.py \
    -e ${EVAL_METHOD} \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -f ${MATRIX_EVAL_DATA_PATH} \
    -t ${TOKENIZER_PATH} \
    -b ${BERT_SOCRE_EVAL_MODEL} \
    -w ${BLEU_WEIGHT}

