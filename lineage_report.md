# Lineage Report (Milestone 3)

## Overview
This repository implements an automated ML workflow using:
- Airflow orchestration (preprocess → train → validate → register)
- MLflow tracking + model registry
- GitHub Actions CI quality gate (fails if accuracy/F1 regress)

Goal: ensure every registered model is traceable to exact code, data artifact, hyperparameters, and metrics.

## Experiment Setup
Dataset: Iris dataset (scikit-learn), exported as CSV during preprocessing.  
Model: Logistic Regression (scikit-learn).  
Tuned hyperparameter: C (inverse regularization strength).  
Tracked metrics: accuracy, macro-F1.

Lineage metadata logged:
- data_hash (SHA256 of the training CSV)
- model_hash (SHA256 of the saved model artifact)
- lineage.json artifact per run
- parameters + metrics in MLflow

## Run Comparison
Run comparisons are exported in `run_comparison.csv` (5+ runs with different C values).

Selection criteria:
1) Pass quality gate thresholds (accuracy/F1)
2) Highest accuracy/F1 among candidates
3) Prefer stable performance across hyperparameter changes

Production candidate:
- The run with best accuracy/F1 that passes validation thresholds.

## Registry Staging Workflow
Promotion path:
- None → Staging after successful Airflow pipeline completion (validation must pass)
- Staging → Production is a governance step (can be automated later with stricter gates)

Each model version includes description and tags tying it to the MLflow run.

## Risks & Monitoring
Data drift:
- Monitor feature distribution drift (PSI/KS)
- Track changes in data_hash and summary stats

Performance drift:
- Periodic evaluation on fresh labeled data
- Alert if accuracy/F1 drops below thresholds

Latency:
- Monitor mean + p95 inference latency
- Alert on regressions

## Rollback Procedure
If Production model degrades:
1) Identify last good model version in MLflow Registry
2) Transition that version back to Production
3) Demote the failing version to Staging/Archived
4) Re-run pipeline with new hyperparameters if needed
