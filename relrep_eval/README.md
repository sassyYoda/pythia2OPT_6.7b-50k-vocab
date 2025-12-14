# Relative Representation Evaluation Suite

This directory contains comprehensive evaluation scripts comparing the relative representation (relrep) alignment method with the vanilla TokAlign method.

## Directory Structure

```
relrep_eval/
├── compare_alignments.py              # Direct alignment quality metrics
├── analyze_by_word_type.py            # Word type analysis
├── analyze_semantic_preservation.py   # Semantic similarity preservation
├── analyze_mapping_patterns.py       # Many-to-one mapping patterns
├── analyze_gold_agreement.py         # Gold label agreement breakdown
├── analyze_geometric_distortion.py    # Embedding space distortion
├── analyze_anchor_quality.py         # Anchor quality analysis (relrep only)
└── compare_bleu_detailed.py          # BLEU score comparison
```

## Quick Start

Run all evaluations:
```bash
bash script/run_all_relrep_evals.sh
```

Or run individual evaluations:
```bash
python relrep_eval/compare_alignments.py
python relrep_eval/analyze_by_word_type.py
# ... etc
```

## Evaluation Scripts

### 1. Direct Alignment Quality Metrics (`compare_alignments.py`)
- Exact agreement between relrep and vanilla methods
- Agreement with gold mappings
- Visualizations: bar charts, Venn diagrams, confusion matrices

**Outputs:**
- `data/pythia2opt-6.7b/alignment_comparison.json`
- `figures/relrep-vs-origin/agreement_plot.png`
- `figures/relrep-vs-origin/venn_diagram.png`
- `figures/relrep-vs-origin/agreement_confusion_matrix.png`

### 2. Word Type Analysis (`analyze_by_word_type.py`)
- Agreement rates by token characteristics:
  - Subword vs full word
  - Short vs long tokens
  - Special vs regular tokens

**Outputs:**
- `data/pythia2opt-6.7b/word_type_metrics.json`
- `figures/relrep-vs-origin/word_type_analysis.png`

### 3. Semantic Similarity Preservation (`analyze_semantic_preservation.py`)
- Analyzes how well semantic similarity is preserved after alignment
- Uses similar word pairs (from file or WordNet)

**Outputs:**
- `data/pythia2opt-6.7b/semantic_preservation_metrics.json`
- `figures/relrep-vs-origin/semantic_preservation.png`

**Input:** `data/word_lists/similar_pairs.txt` (optional)

### 4. Mapping Pattern Analysis (`analyze_mapping_patterns.py`)
- Distribution of targets per source
- Hub tokens (most-mapped-to sources)
- Gini coefficient for mapping inequality

**Outputs:**
- `data/pythia2opt-6.7b/mapping_stats.json`
- `data/pythia2opt-6.7b/hub_tokens.txt`
- `figures/relrep-vs-origin/mapping_patterns.png`
- `figures/relrep-vs-origin/hub_tokens.png`

### 5. Gold Agreement Breakdown (`analyze_gold_agreement.py`)
- Detailed analysis of agreement with gold mappings
- Confusion matrix and Venn diagram visualizations

**Outputs:**
- `data/pythia2opt-6.7b/gold_agreement_stats.json`
- `figures/relrep-vs-origin/gold_agreement_matrix.png`
- `figures/relrep-vs-origin/gold_agreement_venn.png`

### 6. Geometric Distortion Analysis (`analyze_geometric_distortion.py`)
- Distance preservation analysis
- Kruskal's stress metric
- Distortion distributions

**Outputs:**
- `data/pythia2opt-6.7b/distortion_metrics.json`
- `figures/relrep-vs-origin/geometric_distortion.png`

### 7. Anchor Quality Analysis (`analyze_anchor_quality.py`)
- Coverage analysis (PCA variance)
- Anchor diversity
- Informativeness (if relative representations available)

**Outputs:**
- `data/pythia2opt-6.7b/anchor_metrics.json`
- `figures/relrep-vs-origin/anchor_quality.png`

**Note:** Requires `data/pythia2opt-6.7b/align_matrix_relrep_anchors.json`
(Run `bash script/extract_anchors_relrep.sh` first)

### 8. BLEU Score Comparison (`compare_bleu_detailed.py`)
- Compares BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
- Grouped bar chart visualization

**Outputs:**
- `data/pythia2opt-6.7b/bleu_metrics.json`
- `figures/relrep-vs-origin/bleu_comparison.png`

## Dependencies

Required Python packages:
- numpy
- matplotlib
- scipy
- transformers
- scikit-learn (for PCA in anchor analysis)

Optional packages:
- matplotlib-venn (for Venn diagrams)
- nltk (for WordNet semantic pairs)

Install optional dependencies:
```bash
pip install matplotlib-venn nltk
python -c "import nltk; nltk.download('wordnet')"
```

## Input Files Required

All scripts expect these files to exist:
- `data/pythia2opt-6.7b/align_matrix_relrep.json` (relrep alignment matrix)
- `data/pythia2opt-6.7b/align_matrix.json` (vanilla alignment matrix)
- `data/Vocab_count/opt-6.7b2pythia.json` (gold mappings)
- `data/vec-mix-pythia.txt` (source GloVe vectors)
- `data/vec-mix-opt-6.7b.txt` (target GloVe vectors)
- `data/pretrain-dataset/pythia-2-opt-6.7b-glove-eval-mix` (evaluation data)

Optional:
- `data/pythia2opt-6.7b/align_matrix_relrep_anchors.json` (for anchor analysis)
- `data/word_lists/similar_pairs.txt` (for semantic preservation)

## Output Locations

All metrics are saved to: `data/pythia2opt-6.7b/`
All figures are saved to: `figures/relrep-vs-origin/`

## Customization

All scripts accept command-line arguments to customize paths. Use `--help` to see options:
```bash
python relrep_eval/compare_alignments.py --help
```

