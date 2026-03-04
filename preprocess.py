import argparse
import os
import pandas as pd
from sklearn.datasets import load_iris


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--run-suffix", default="manual")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    out_path = os.path.join(args.outdir, f"preprocessed_{args.run_suffix}.csv")
    df.to_csv(out_path, index=False)

    print(out_path)


if __name__ == "__main__":
    main()
