from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_dataset_csv(X: np.ndarray, y: np.ndarray, out_csv: Path) -> None:
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    df.to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations for solver")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--outdir", required=True, help="Output directory (Airflow passes artifacts/runs/<date>)")
    parser.add_argument("--experiment", default="milestone3", help="MLflow experiment name")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    artifacts_root = repo_root / "artifacts"
    ensure_dir(artifacts_root)

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.2,
        random_state=args.seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    data_csv = outdir / "train_data.csv"
    save_dataset_csv(X_train, y_train, data_csv)

    data_hash = sha256_file(data_csv)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        (outdir / "run_id.txt").write_text(run_id)

        mlflow.log_params(
            {
                "C": args.C,
                "max_iter": args.max_iter,
                "seed": args.seed,
                "data_path": str(data_csv),
                "tracking_uri": tracking_uri,
            }
        )
        mlflow.set_tag("data_hash", data_hash)

        if os.environ.get("GITHUB_SHA"):
            mlflow.set_tag("git_commit", os.environ["GITHUB_SHA"])
        mlflow.set_tag("runner", "local" if not os.environ.get("GITHUB_ACTIONS") else "github-actions")

        model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        model_path = outdir / "model.pkl"
        joblib.dump(model, model_path)

        model_hash = sha256_file(model_path)
        mlflow.set_tag("model_hash", model_hash)

        mlflow.log_artifact(str(data_csv), artifact_path="data")
        mlflow.log_artifact(str(model_path), artifact_path="model_file")

        mlflow.sklearn.log_model(model, artifact_path="model")

        summary = outdir / "summary.txt"
        summary.write_text(
            f"run_id={run_id}\n"
            f"accuracy={acc}\n"
            f"f1={f1}\n"
            f"data_hash={data_hash}\n"
            f"model_hash={model_hash}\n"
        )
        mlflow.log_artifact(str(summary), artifact_path="reports")

        print(f"RUN_ID={run_id}")
        print(f"accuracy={acc:.4f} f1={f1:.4f}")
        print(f"model_hash={model_hash}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
