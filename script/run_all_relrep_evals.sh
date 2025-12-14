#!/bin/sh

# Master script to run all relative representation evaluations

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd ${MAIN_DIR}

printf "\n"
printf "========================================\n"
printf "Running All RelRep Evaluations\n"
printf "========================================\n"
printf "\n"

# 1. Direct Alignment Quality Metrics
printf "1. Running Direct Alignment Quality Metrics...\n"
python relrep_eval/compare_alignments.py
printf "\n"

# 2. Word Type Analysis
printf "2. Running Word Type Analysis...\n"
python relrep_eval/analyze_by_word_type.py
printf "\n"

# 3. Semantic Preservation
printf "3. Running Semantic Preservation Analysis...\n"
python relrep_eval/analyze_semantic_preservation.py
printf "\n"

# 4. Mapping Patterns
printf "4. Running Mapping Pattern Analysis...\n"
python relrep_eval/analyze_mapping_patterns.py
printf "\n"

# 5. Gold Agreement
printf "5. Running Gold Agreement Analysis...\n"
python relrep_eval/analyze_gold_agreement.py
printf "\n"

# 6. Geometric Distortion
printf "6. Running Geometric Distortion Analysis...\n"
python relrep_eval/analyze_geometric_distortion.py
printf "\n"

# 7. BLEU Comparison
printf "7. Running BLEU Comparison...\n"
python relrep_eval/compare_bleu_detailed.py
printf "\n"

# 8. Anchor Quality (if anchors file exists)
if [ -f "./data/pythia2opt-6.7b/align_matrix_relrep_anchors.json" ]; then
    printf "8. Running Anchor Quality Analysis...\n"
    python relrep_eval/analyze_anchor_quality.py
    printf "\n"
else
    printf "8. Skipping Anchor Quality Analysis (anchors file not found)\n"
    printf "   Run: bash script/extract_anchors_relrep.sh\n"
    printf "\n"
fi

printf "========================================\n"
printf "All Evaluations Complete!\n"
printf "========================================\n"
printf "\n"
printf "Results saved to:\n"
printf "  - Metrics: ./data/pythia2opt-6.7b/\n"
printf "  - Figures: ./figures/relrep-vs-origin/\n"
printf "\n"

