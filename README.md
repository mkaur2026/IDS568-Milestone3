# IDS 568 – Milestone 3

## Airflow Pipeline + MLflow Tracking + CI Quality Gate

## Overview

This project implements an end-to-end MLOps pipeline for training, tracking, validating, and registering a machine learning model.

The pipeline integrates:

* **MLflow** for experiment tracking and model registry  
* **Apache Airflow** for pipeline orchestration  
* **GitHub Actions** for CI validation and quality gates  

The goal of this milestone is to demonstrate how a machine learning workflow can be automated, validated, and tracked in a reproducible and production-style pipeline.

---

## Repository Structure

```
.
├── dags/
│   └── train_pipeline.py
├── data/
├── artifacts/
├── src/
├── preprocess.py
├── train.py
├── model_validation.py
├── register_model.py
├── compare_runs.py
├── run_comparison.csv
├── lineage_report.md
├── requirements.txt
├── requirements-airflow.txt
├── mlflow.db
└── .github/workflows/
    └── train_and_validate.yml
```

---

## Machine Learning Workflow

The pipeline contains four major stages.

### 1. Data Preprocessing

`preprocess.py`

This script prepares the dataset used for training.

Steps performed:

- Loads dataset
- Removes duplicates
- Saves processed dataset as CSV artifact

Example output:

```
artifacts/preprocessed_<run_id>.csv
```

---

### 2. Model Training

`train.py`

This script trains a Logistic Regression model and logs experiment metadata to MLflow.

Tracked information includes:

- Hyperparameters
- Accuracy
- F1 score
- Model artifact
- Training dataset artifact
- Model hash
- Data hash

This ensures full reproducibility of every experiment run.

---

### 3. Quality Gate Validation

`model_validation.py`

Before a model is accepted into the pipeline, it must pass validation thresholds.

Validation rules:

```
Accuracy >= 0.90
F1 Score >= 0.85
```

If the model does not meet these requirements, the pipeline fails.

This acts as a **CI quality gate**.

---

### 4. Model Registration

`register_model.py`

If validation succeeds, the model is registered in the MLflow Model Registry.

Model lifecycle stages:

```
None → Staging → Production
```

In this milestone the pipeline registers models into **Staging**.

---

## Experiment Tracking

All experiments are tracked using **MLflow**.

Tracking backend:

```
mlflow.db
```

Artifact storage:

```
mlruns_artifacts/
```

Each MLflow run logs:

- hyperparameters
- evaluation metrics
- training dataset artifact
- model artifact
- lineage metadata

---

## Pipeline Orchestration

The pipeline is orchestrated using **Apache Airflow**.

DAG file:

```
dags/train_pipeline.py
```

Pipeline flow:

```
preprocess_data
      ↓
train_model
      ↓
validate_model
      ↓
register_model
```

The DAG includes:

- retries
- retry delays
- failure callbacks
- task dependency ordering

---

## CI/CD Pipeline

Continuous integration is implemented with **GitHub Actions**.

Workflow file:

```
.github/workflows/train_and_validate.yml
```

The workflow performs:

1. Install project dependencies
2. Run model training
3. Execute validation quality gate
4. Upload MLflow tracking database as CI artifact

This ensures every commit is automatically validated.

---

## Lineage Tracking

Lineage metadata is stored in:

```
artifacts/lineage.json
```

The file records:

```
run_id
hyperparameters
metrics
model_hash
data_hash
tracking metadata
```

A detailed explanation of lineage tracking is provided in:

```
lineage_report.md
```

---

## Run Comparison

Multiple training runs were executed with different hyperparameters.

Results are stored in:

```
run_comparison.csv
```

This file compares metrics across runs to identify the best-performing model.

---

## Local Setup

Install dependencies:

```
pip install -r requirements.txt
pip install -r requirements-airflow.txt
```

---

## Manual Pipeline Execution

The pipeline can be executed manually without Airflow:

```
DATA_PATH=$(python preprocess.py --outdir artifacts --run-suffix test)

RUN_ID=$(python train.py \
  --tracking-uri sqlite:///mlflow.db \
  --experiment milestone3 \
  --C 1.0 \
  --data-path "$DATA_PATH")

python model_validation.py \
  --tracking-uri sqlite:///mlflow.db \
  --run-id "$RUN_ID" \
  --min-accuracy 0.90 \
  --min-f1 0.85

python register_model.py \
  --tracking-uri sqlite:///mlflow.db \
  --run-id "$RUN_ID" \
  --model-name milestone3-model \
  --stage Staging
```

---

## Running the Airflow Pipeline

Initialize Airflow:

```
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
```

Create admin user:

```
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

Start services:

```
airflow webserver --port 8080
```

In a second terminal:

```
airflow scheduler
```

Then trigger the **train_pipeline** DAG in the Airflow UI.

---

## Automated Sanity Checks

Verify required files exist:

```
ls dags/train_pipeline.py \
   model_validation.py \
   train.py \
   register_model.py \
   preprocess.py \
   requirements.txt \
   .github/workflows/train_and_validate.yml \
   lineage_report.md
```

---

## Key MLOps Features Demonstrated

✔ MLflow experiment tracking  
✔ Airflow workflow orchestration  
✔ CI/CD pipeline with GitHub Actions  
✔ Model validation quality gates  
✔ Experiment lineage tracking  
✔ Automated model registration

---
