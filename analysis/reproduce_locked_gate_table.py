"""Reconstruct the locked G/D gate comparison from published participant summaries."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def reconstruct(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {(str(person), str(seed)) for person in (13, 14, 15) for seed in (42, 43, 44)}
    if len(rows) != 9 or {(r["subject"], r["seed"]) for r in rows} != expected:
        raise ValueError("Expected one row per locked participant and matched seed.")
    by_person = {}
    for row in rows:
        g, d = float(row["G_mae_bpm"]), float(row["D_mae_bpm"])
        diff = d - g
        if not math.isclose(diff, float(row["D_minus_G_bpm"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Archived paired difference disagrees with D minus G.")
        by_person.setdefault(row["subject"], {})[row["seed"]] = diff
    means = {person: statistics.mean(cells.values()) for person, cells in by_person.items()}
    effects = list(means.values())
    delta = statistics.mean(effects)
    sd = statistics.stdev(effects)
    # Two-sided 95% Student interval, df=2, assuming independent normal person effects.
    margin = 4.302652729749464 * sd / math.sqrt(3)
    return {
        "unit": "bpm", "people": 3, "matched_seeds_per_person": 3,
        "G_equal_person_MAE": statistics.mean(float(row["G_mae_bpm"]) for row in rows),
        "D_equal_person_MAE": statistics.mean(float(row["D_mae_bpm"]) for row in rows),
        "D_minus_G_by_person_and_seed": by_person, "person_means": means,
        "D_minus_G_equal_person": delta, "observed_person_range": [min(effects), max(effects)],
        "person_sample_SD": sd, "normal_model_t95_sensitivity": [delta-margin, delta+margin],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "audit/20261118/locked_final_D_minus_G_nine_cells.csv"
    parser.add_argument("summary", type=Path, nargs="?", default=default)
    print(json.dumps(reconstruct(parser.parse_args().summary), indent=2))
