from datetime import datetime, timedelta
import os
import subprocess
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


def on_failure_callback(context):
    ti = context.get("task_instance")
    print(f"[FAILURE] task={ti.task_id} dag={ti.dag_id} run_id={ti.run_id}")


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def preprocess_data(**context):
    """
    Idempotent: creates a dated run folder and stores path in XCom.
    """
    run_date = context["ds"]  # e.g. 2026-03-03
    run_dir = ARTIFACTS_DIR / "runs" / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "README.txt").write_text("Run folder for this DAG date.\n")

    # Pass run_dir to next task
    context["ti"].xcom_push(key="run_dir", value=str(run_dir))


def train_model(**context):
    run_dir = context["ti"].xcom_pull(key="run_dir", task_ids="preprocess_data")
    if not run_dir:
        raise RuntimeError("Missing run_dir from preprocess_data")

    # Example hyperparam (you can change later)
    # Make sure train.py writes model + logs metrics to MLflow
    cmd = [
        "python",
        str(REPO_ROOT / "train.py"),
        "--C",
        "1.0",
        "--outdir",
        run_dir,
    ]
    subprocess.run(cmd, check=True)

    # train.py should write the run_id into a file OR print it.
    # simplest: have train.py write artifacts/latest_run_id.txt
    run_id_file = Path(run_dir) / "run_id.txt"
    if not run_id_file.exists():
        raise RuntimeError(f"train.py did not write {run_id_file}")
    run_id = run_id_file.read_text().strip()

    context["ti"].xcom_push(key="run_id", value=run_id)


def register_model(**context):
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    if not run_id:
        raise RuntimeError("Missing run_id from train_model")

    cmd = [
        "python",
        str(REPO_ROOT / "register_model.py"),
        "--tracking-uri",
        "sqlite:///mlflow.db",
        "--run-id",
        run_id,
        "--model-name",
        "milestone3-model",
        "--stage",
        "Staging",
        "--description",
        f"Registered by Airflow DAG run_id={run_id}",
    ]
    subprocess.run(cmd, check=True)


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        provide_context=True,
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )
    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        provide_context=True,
    )

    preprocess >> train >> register
