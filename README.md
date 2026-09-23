# Beyond Accuracy — reproduction code

Code, data and results for the manuscript *"Beyond Accuracy: A Multi-Axis Framework for
Statistical Model Comparison, Calibration, and Explanation Stability in Medical Machine
Learning"* (M. Mohammed, Northern Technical University).

## Layout

All files are in the repository root (flat layout): data files, scripts (`*.py`), raw results (`*.csv`, `*.npz`, `*.json`), summary tables (`tbl_*.csv`) and figures (`fig*.png`).

## Data (SHA-256 in `SHA256SUMS`)

| File | Source | Notes |
|---|---|---|
| (loaded from scikit-learn) | UCI Breast Cancer Wisconsin (Diagnostic), doi:10.24432/C5DW2B | 569 cases, 212 malignant (positive) |
| `pima-indians-diabetes.csv` | Pima Indians Diabetes (Smith et al., 1988), standard 768-case file | zeros in glucose, BP, skin, insulin, BMI treated as missing |
| `processed.cleveland.data` | UCI Heart Disease, **original processed Cleveland file** (Detrano et al., 1989), doi:10.24432/C52P4X | `num > 0` = disease (139/303); `?` in `ca` (4) and `thal` (2) = missing |
| `diabetic_data.csv` | UCI Diabetes 130-US Hospitals 1999–2008 (Strack et al., 2014), doi:10.24432/C5230J | first encounter per patient, deaths/hospice removed, outcome = readmission < 30 days; stratified subsample of 10,000 patients (seed 2026) |

The widely circulated Kaggle `heart.csv` is **not** used: its `target` is inverted relative
to UCI (`target = 1` corresponds to UCI `num = 0`) and it contains invalid codes
(`ca = 4`, `thal = 0`).

## Environment

Python 3.11; pinned versions in `requirements.txt`
(scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0, SciPy 1.17.1, NumPy, pandas 3.0.2,
statsmodels, PyMC 5.28.5). CPU only.

```bash
pip install -r requirements.txt
bash run_all.sh          # approx. 3-4 h on a 2-core CPU
```

## Seeds

| Purpose | Seed |
|---|---|
| Diabetes-130 subsample | 2026 |
| 5 × 10 repeated stratified CV (all models share partitions) | 42 |
| Model seed in benchmark / nested CV | 0 |
| Nested CV outer / inner folds | 7 / 11 |
| SHAP seed axis | model seeds 0–29 |
| SHAP bootstrap axis | resample seeds 1000–1029, model seed 0 |
| SHAP evaluation subsample (Diabetes-130) | 5 |
| Permutation-importance split / D130 test subsample | 3 / 4 |
| Permutation designs | model seeds 0–R-1, permutation seeds 0–R-1 |
| PyMC sampling | 0 |

## Scripts

| Script | Output |
|---|---|
| `data.py` | dataset loaders (run directly to print Table 1 and Pima missingness) |
| `exp_benchmark.py` | determinism check, 5×10 CV metrics, calibration slope/intercept, OOF predictions |
| `exp_nested.py` | nested CV, default vs tuned on identical outer folds |
| `exp_shap.py` | SHAP seed/bootstrap stability, collinearity diagnostics |
| `exp_permutation.py` | permutation-importance stability with model/permutation randomness separated |
| `exp_pima_imputation.py` | imputation sensitivity (median / iterative / drop) |
| `analysis.py` | corrected resampled t-tests, Bayesian correlated and hierarchical tests, tables |
| `figures.py`, `fig_framework.py` | figures |

## Licence

Code: MIT. Datasets: see the licences of the original sources (UCI datasets CC BY 4.0).
