import glob
import os

import pandas as pd


def main():
    paths = sorted(glob.glob("logs/cifar10_resnet18_seed*.csv"))

    if len(paths) == 0:
        raise FileNotFoundError("No csv logs found in logs/")

    rows = []

    for path in paths:
        df = pd.read_csv(path)
        last = df.iloc[-1]

        rows.append(
            {
                "file": os.path.basename(path),
                "seed": int(last["seed"]),
                "best_test_acc": float(last["best_test_acc"]),
                "final_test_acc": float(last["test_acc"]),
            }
        )

    result = pd.DataFrame(rows)

    print(result)
    print()
    print("best_test_acc mean:", result["best_test_acc"].mean())
    print("best_test_acc std:", result["best_test_acc"].std())
    print("final_test_acc mean:", result["final_test_acc"].mean())
    print("final_test_acc std:", result["final_test_acc"].std())


if __name__ == "__main__":
    main()