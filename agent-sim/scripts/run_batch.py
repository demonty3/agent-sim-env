#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List

from sim.evaluation import summarize_outcomes
from sim.models import Agent, Issue
from sim.policies import LinearConcessionPolicy, ThresholdAcceptancePolicy
from sim.protocol import run_negotiation


def main() -> None:
    p = argparse.ArgumentParser(description="Run a batch of negotiations and print summary metrics")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    issues = [Issue("price", 0.0, 1.0, 0.7), Issue("warranty", 0.0, 1.0, 0.3)]
    outcomes = []
    for _ in range(args.runs):
        a = Agent("Alice", issues, 0.2, LinearConcessionPolicy(), ThresholdAcceptancePolicy(0.6))
        b = Agent("Bob", issues, 0.3, LinearConcessionPolicy(), ThresholdAcceptancePolicy(0.55))
        outcomes.append(run_negotiation(a, b, max_rounds=7))

    summary = summarize_outcomes(outcomes)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

