#!/usr/bin/env bash
# Additional analyses added in response to the reviewer comments (R2 revision).
set -e
python exp_rev_calibration.py                 # comment 2: calibration of tuned models
python exp_rev_shap.py tuned insample         # comment 1: SHAP stability, tuned configurations
python exp_rev_threshold.py                   # comment 3: Youden-threshold sensitivity
python exp_rev_shap.py default heldout BC PIMA HEART   # comment 4: held-out SHAP evaluation
python exp_rev_shap.py tuned heldout BC PIMA HEART
python exp_rev_shap.py default heldout D130
python exp_rev_shap.py tuned heldout D130
