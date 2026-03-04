from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def on_failure_callback(context):
    print("Task failed:", context.get("task_instance").task_id)

def preprocess_data():
    print("preprocess_data")

def train_model():
    print("train_model")

def register_model():
    print("register_model")

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
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    register = PythonOperator(task_id="register_model", python_callable=register_model)

    preprocess >> train >> register
