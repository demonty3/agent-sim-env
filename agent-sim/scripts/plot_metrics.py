#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from statistics import mean


def main() -> None:
    p = argparse.ArgumentParser(description="Read a CSV of outcomes and print quick stats")
    p.add_argument("path", help="CSV path with columns: agreement,rounds,utility_a,utility_b")
    args = p.parse_args()

    rows = []
    with open(args.path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No data.")
        return

    def f(col: str):
        return [float(r[col]) for r in rows]

    agreements = [1.0 if r["agreement"].lower() in ("1", "true", "yes") else 0.0 for r in rows]
    print("count:", len(rows))
    print("agreement_rate:", round(mean(agreements), 4))
    print("avg_rounds:", round(mean(f("rounds")), 4))
    print("avg_utility_a:", round(mean(f("utility_a")), 4))
    print("avg_utility_b:", round(mean(f("utility_b")), 4))


if __name__ == "__main__":
    main()

