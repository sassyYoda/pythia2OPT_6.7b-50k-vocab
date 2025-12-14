#!/bin/sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

# Use relative representation alignment matrix instead of regular one
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/align_matrix_relrep.json"
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/align_matrix_relrep_demo.json"

export MODLE_PATH1="EleutherAI/pythia-1b"

export TOKENIZER_PATH2="facebook/opt-6.7b"

# Output path for relative representation initialized model
export OUTPUT_PATH="${MAIN_DIR}/data/pythia2opt-6.7b/TokAlign-RelRep-Init-1B"

printf "\n### Initializing model with relative representation alignment matrix ###\n\n"
printf "Alignment matrix: ${TGT_ID_2_SRC_ID_RES_PATH}\n"
printf "Source model: ${MODLE_PATH1}\n"
printf "Target tokenizer: ${TOKENIZER_PATH2}\n"
printf "Output path: ${OUTPUT_PATH}\n\n"

python src/convert.py \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -s ${MODLE_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${OUTPUT_PATH}

