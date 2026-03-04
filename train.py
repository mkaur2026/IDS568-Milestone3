import argparse
import hashlib
import json
import os
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def ensure_local_experiment(experiment_name: str, artifact_dir: str) -> None:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=f"file:{os.path.abspath(artifact_dir)}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "milestone3"))
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--artifact-dir", default="mlruns_artifacts")
    parser.add_argument("--data-path", default="")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    ensure_local_experiment(args.experiment, args.artifact_dir)
    mlflow.set_experiment(args.experiment)

    # Load data (from preprocess output if provided)
    if args.data_path and os.path.exists(args.data_path):
        df = pd.read_csv(args.data_path)
        data_path = args.data_path
    else:
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        data_path = os.path.join(args.outdir, "train_data.csv")
        df.to_csv(data_path, index=False)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        model = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        model_path = os.path.join(args.outdir, "model.pkl")
        joblib.dump(model, model_path)

        model_hash = file_hash(model_path)
        data_hash = file_hash(data_path)

        mlflow.log_params(
            {
                "C": args.C,
                "max_iter": args.max_iter,
                "data_path": os.path.abspath(data_path),
                "tracking_uri": args.tracking_uri,
            }
        )
        mlflow.log_metrics({"accuracy": float(acc), "f1": float(f1)})

        mlflow.log_artifact(model_path, artifact_path="model_artifacts")
        mlflow.log_artifact(data_path, artifact_path="data_artifacts")

        mlflow.set_tag("model_hash", model_hash)
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("trained_at_utc", datetime.utcnow().isoformat())

        lineage = {
            "run_id": run_id,
            "params": {"C": args.C, "max_iter": args.max_iter},
            "metrics": {"accuracy": float(acc), "f1": float(f1)},
            "hashes": {"model_hash": model_hash, "data_hash": data_hash},
            "tracking": {"tracking_uri": args.tracking_uri, "experiment": args.experiment},
        }
        lineage_path = os.path.join(args.outdir, "lineage.json")
        with open(lineage_path, "w") as f:
            json.dump(lineage, f, indent=2)
        mlflow.log_artifact(lineage_path, artifact_path="lineage")

        # IMPORTANT: print run_id so Airflow/CI can capture it
        print(run_id)


if __name__ == "__main__":
    main()
