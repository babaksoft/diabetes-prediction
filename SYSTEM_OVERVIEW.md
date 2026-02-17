# ML System Overview — Diabetes Risk Prediction

## Problem
Early identification of diabetes risk requires different decision
strategies at different stages of care.

## Solution
A single probabilistic classifier supports two operational modes
through threshold-based policies:
- Triage (high recall)
- Follow-up (high precision)

## Data
- 95,964 cleaned records
- Imbalanced binary target
- Strict train/validation/test separation
- All splits versioned and frozen

## Model
- HistGradientBoostingClassifier
- Optimized for PR-AUC
- Thresholds selected on validation set

## Decision Policies
Triage:
- Recall ≥ 0.96
- Precision ≥ 0.25

Balanced:
- F1 ≥ 0.68

## Validation
- Cross-validation for model selection
- Threshold analysis on validation set
- Final evaluation on held-out test set

## Deployment
- FastAPI inference service
- Streamlit UI
- Dockerized local deployment
- Model loaded from exported MLflow artifacts

## Trust & Reproducibility
- MLflow for experiment tracking
- DVC for data versioning
- Explicit business constraints
