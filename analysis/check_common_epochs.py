"""Check the published inner-validation table against the extracted public histories."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def check(bundle, table):
    results = {}
    overlap = []
    n_paired_epochs = 0
    for objective in ("logcosh", "MSE", "MAE", "Huber5"):
        for fold in (1, 2, 3):
            for seed in (42, 43, 44):
                histories = []
                for method in ("D", "G"):
                    run = f"f{fold}_s{seed}_{objective}_{method}"
                    path = bundle / "evidence/development_cv" / run / "history.json"
                    histories.append(json.loads(path.read_text(encoding="utf-8")))
                shared = min(map(len, histories))
                overlap.append(shared)
                n_paired_epochs += shared
                for index, (d, g) in enumerate(zip(*histories), 1):
                    if d["epoch"] != index or g["epoch"] != index or d["updates"] != g["updates"]:
                        raise ValueError(f"Unmatched epoch or optimizer step in {run}")
                    results.setdefault((objective, fold, index), []).append(
                        d["inner_equal_person_mae"] - g["inner_equal_person_mae"])
    shared_last = min(overlap)
    expected = {(obj, fold, epoch) for obj, fold, epoch in results if epoch in (10, 20, shared_last)}
    with table.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = row["objective"], int(row["fold"]), int(row["epoch"])
            expected.remove(key)
            values = results[key]
            checks = [(mean(values), float(row["D_minus_G_inner_mean_bpm"])),
                      (median(values), float(row["D_minus_G_inner_median_bpm"])),
                      (sum(v < 0 for v in values), int(row["D_lower_count"])),
                      (len(values), int(row["n_seeds"]))]
            if any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in checks):
                raise ValueError(f"Published values differ for {key}")
    if expected:
        raise ValueError(f"Missing rows: {expected}")
    print(f"Verified {len(overlap)} pairs and {n_paired_epochs} paired inner epochs. Last common epoch: {shared_last}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Root of the extracted reproducibility ZIP")
    parser.add_argument("--table", type=Path, default=Path(__file__).parent / "20261119/same_epoch_inner_within_fold_summary.csv")
    args = parser.parse_args()
    check(args.bundle, args.table)
