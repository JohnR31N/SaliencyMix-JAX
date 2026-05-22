import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.dataset and args.model:
        pattern = f"{args.dataset}_{args.model}_seed*.csv"
    elif args.dataset:
        pattern = f"{args.dataset}_*_seed*.csv"
    elif args.model:
        pattern = f"*_{args.model}_seed*.csv"
    else:
        pattern = "*.csv"

    paths = sorted(glob.glob(os.path.join(args.log_dir, pattern)))

    if len(paths) == 0:
        raise FileNotFoundError(f"No csv logs found with pattern: {os.path.join(args.log_dir, pattern)}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        last = df.iloc[-1]
        rows.append(
            {
                "file": os.path.basename(path),
                "dataset": last.get("dataset", "unknown"),
                "model": last.get("model", "unknown"),
                "seed": int(last["seed"]),
                "best_test_acc": float(last["best_test_acc"]),
                "best_test_error": 100.0 - float(last["best_test_acc"]),
                "final_test_acc": float(last["test_acc"]),
                "final_test_error": float(last.get("test_error", 100.0 - float(last["test_acc"]))),
                "total_time_sec": float(last.get("total_time_sec", 0.0)),
            }
        )

    result = pd.DataFrame(rows)
    print(result)
    print()

    summary = result.groupby(["dataset", "model"]).agg(
        best_test_acc_mean=("best_test_acc", "mean"),
        best_test_acc_std=("best_test_acc", "std"),
        best_test_error_mean=("best_test_error", "mean"),
        best_test_error_std=("best_test_error", "std"),
        final_test_acc_mean=("final_test_acc", "mean"),
        final_test_acc_std=("final_test_acc", "std"),
        total_time_sec_mean=("total_time_sec", "mean"),
    )
    print(summary)


if __name__ == "__main__":
    main()
