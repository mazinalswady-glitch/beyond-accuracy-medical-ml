#!/usr/bin/env bash
# Full reproduction of all results (approx. 3-4 h on a 2-core CPU; dominated by the
# Diabetes-130 kernel-SVM and MLP fits). Run from the repository root.
set -e
mkdir -p results figures
python src/exp_benchmark.py all      # E0 determinism + E1 benchmark/calibration
python src/exp_shap.py               # E4 SHAP two-axis stability + collinearity
python src/exp_pima_imputation.py    # E6 Pima imputation sensitivity
python src/exp_nested.py             # E3 nested CV (default vs tuned, paired)
python src/exp_permutation.py        # E5 permutation-importance source separation
python src/analysis.py               # statistics, tables
python src/figures.py                # figures
