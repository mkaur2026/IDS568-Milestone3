from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

REPO_ROOT = Path(__file__).resolve().parents[1]


def on_failure_callback(context):
    ti = context.get("task_instance")
    print(f"[FAILURE] task_id={ti.task_id} dag_id={ti.dag_id} run_id={context.get('run_id')}")


def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    return result.stdout.strip()


def preprocess_data(ds_nodash: str, **_):
    out = run_cmd(["python", "preprocess.py", "--outdir", "artifacts", "--run-suffix", ds_nodash])
    return out


def train_model(ti, **_):
    data_path = ti.xcom_pull(task_ids="preprocess_data")
    run_id = run_cmd(
        [
            "python",
            "train.py",
            "--tracking-uri",
            "sqlite:///mlflow.db",
            "--experiment",
            "milestone3",
            "--C",
            "1.0",
            "--max-iter",
            "200",
            "--outdir",
            "artifacts",
            "--data-path",
            data_path,
        ]
    )
    return run_id


def validate_model(ti, **_):
    run_id = ti.xcom_pull(task_ids="train_model")
    run_cmd(
        [
            "python",
            "model_validation.py",
            "--tracking-uri",
            "sqlite:///mlflow.db",
            "--run-id",
            run_id,
            "--min-accuracy",
            "0.90",
            "--min-f1",
            "0.85",
        ]
    )
    return "passed"


def register_model(ti, **_):
    run_id = ti.xcom_pull(task_ids="train_model")
    run_cmd(
        [
            "python",
            "register_model.py",
            "--tracking-uri",
            "sqlite:///mlflow.db",
            "--run-id",
            run_id,
            "--model-name",
            "milestone3-model",
            "--stage",
            "Staging",
            "--description",
            "Registered by Airflow after validation pass",
        ]
    )
    return "registered"


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
    schedule=None,
    catchup=False,
) as dag:
    t1 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        op_kwargs={"ds_nodash": "{{ ds_nodash }}"},
    )

    t2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t3 = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    t4 = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    t1 >> t2 >> t3 >> t4
