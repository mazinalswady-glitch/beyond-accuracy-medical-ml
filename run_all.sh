#!/usr/bin/env bash
# Full reproduction of all results (approx. 3-4 h on a 2-core CPU; dominated by the
# Diabetes-130 kernel-SVM and MLP fits). Run from the repository root.
set -e
python exp_benchmark.py all      # E0 determinism + E1 benchmark/calibration
python exp_shap.py               # E4 SHAP two-axis stability + collinearity
python exp_pima_imputation.py    # E6 Pima imputation sensitivity
python exp_nested.py             # E3 nested CV (default vs tuned, paired)
python exp_permutation.py        # E5 permutation-importance source separation
python analysis.py               # statistics, summary tables (tbl_*.csv)
python figures.py                # figures 2-7
python fig_framework.py          # figure 1
