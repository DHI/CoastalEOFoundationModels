"""Tabulate `results_<stage>.json` records found under a directory.

    python scripts/aggregate_results.py outputs/
    python scripts/aggregate_results.py multirun/ --mean --metrics test/mIoU test/Pixel_Accuracy --csv table.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="directory searched recursively")
    parser.add_argument("--stage", default="test", choices=["fit", "test"])
    parser.add_argument("--metrics", nargs="+", help="metric columns to keep (default: all)")
    parser.add_argument("--mean", action="store_true", help="mean and std over seeds per experiment")
    parser.add_argument("--csv", type=Path, help="also write the table to this CSV file")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob(f"results_{args.stage}.json")):
        record = json.loads(path.read_text())
        rows.append({"experiment": record["experiment"], "protocol": record["protocol"],
                     "seed": record["seed"], **record["metrics"]})
    if not rows:
        raise SystemExit(f"no results_{args.stage}.json under {args.root}")

    table = pd.DataFrame(rows).sort_values(["experiment", "protocol", "seed"])
    metrics = args.metrics or [c for c in table.columns if c not in ("experiment", "protocol", "seed")]
    table = table[["experiment", "protocol", "seed", *metrics]]
    if args.mean:
        grouped = table.groupby(["experiment", "protocol"], dropna=False)
        table = grouped[metrics].agg(["mean", "std"])
        table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
        table.insert(0, "seeds", grouped.size())
        table = table.reset_index()

    print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
