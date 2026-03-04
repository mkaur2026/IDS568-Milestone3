IDS 568 – Milestone 3
Airflow Pipeline + MLflow Tracking + CI Quality Gate
Overview
This project implements an end-to-end MLOps pipeline for training, tracking, validating, and registering a machine learning model.
The pipeline integrates the following tools:
MLflow for experiment tracking and model registry
Apache Airflow for pipeline orchestration
GitHub Actions for CI validation and quality gates
The goal of this milestone is to demonstrate how a machine learning workflow can be automated and monitored using modern MLOps practices.
Repository Structure
.
├── dags/
│   └── train_pipeline.py        # Airflow DAG for training pipeline
│
├── .github/workflows/
│   └── train_and_validate.yml   # CI pipeline
│
├── train.py                     # Model training + MLflow logging
├── compare_runs.py              # Compare MLflow experiment runs
├── model_validation.py          # Quality gate validation
├── register_model.py            # Register model in MLflow registry
│
├── requirements.txt             # Project dependencies
├── run_comparison.csv           # Output from experiment comparison
├── lineage_report.md            # Data and model lineage documentation
└── README.md
ML Pipeline Components
1. Model Training
train.py
This script:
trains a machine learning model
logs parameters and metrics to MLflow
saves artifacts for reproducibility
Example:
python train.py --C 1.0
2. Experiment Comparison
compare_runs.py
This script compares multiple MLflow runs and generates a summary table.
Example:
python compare_runs.py
Output:
run_comparison.csv
3. Model Validation (Quality Gate)
model_validation.py
Implements automated model checks using performance thresholds.
Example:
python model_validation.py --run-id RUN_ID --min-accuracy 0.70 --min-f1 0.60
If the model fails validation, the script exits with a failure code.
4. Model Registration
register_model.py
Registers the trained model in the MLflow Model Registry and moves it to the Staging stage.
Example:
python register_model.py \
--tracking-uri sqlite:///mlflow.db \
--run-id RUN_ID \
--model-name milestone3-model \
--stage Staging
Airflow Pipeline
The Airflow DAG (train_pipeline.py) orchestrates the ML workflow.
Pipeline steps:
preprocess_data
      ↓
train_model
      ↓
register_model
Run the DAG locally:
airflow dags test train_pipeline 2026-03-03
Continuous Integration (CI)
GitHub Actions automatically runs:
model training
validation checks
quality gate enforcement
CI configuration is located in:
.github/workflows/train_and_validate.yml
Lineage Documentation
lineage_report.md documents:
data sources
preprocessing steps
model artifacts
experiment tracking
This ensures reproducibility and transparency of the ML workflow.
Requirements
Install dependencies:
pip install -r requirements.txt
Summary
This project demonstrates an automated machine learning pipeline that includes:
experiment tracking with MLflow
automated validation using quality gates
model registry management
workflow orchestration with Airflow
CI validation using GitHub Actions
This architecture ensures reproducibility, traceability, and automation for ML model development.
