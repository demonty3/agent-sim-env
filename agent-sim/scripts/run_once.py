#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from sim.models import Agent, Issue
from sim.policies import LinearConcessionPolicy, ThresholdAcceptancePolicy
from sim.protocol import run_negotiation


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run without --config or install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agents(cfg: Dict[str, Any]) -> tuple[Agent, Agent, int]:
    issues = [Issue(**d) for d in cfg.get("issues", [])]
    max_rounds = int(cfg.get("max_rounds", 10))

    def mk_agent(block: Dict[str, Any]) -> Agent:
        name = block.get("name", "Agent")
        rv = float(block.get("reservation_value", 0.0))
        # For now we only wire the sample policies from this repo
        cp = LinearConcessionPolicy()
        ap_cfg = block.get("acceptance", {"type": "threshold", "threshold": 0.5})
        ap = ThresholdAcceptancePolicy(float(ap_cfg.get("threshold", 0.5)))
        return Agent(name=name, issues=issues, reservation_value=rv, concession_policy=cp, acceptance_policy=ap)

    a = mk_agent(cfg.get("agent_a", {}))
    b = mk_agent(cfg.get("agent_b", {}))
    return a, b, max_rounds


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single negotiation simulation")
    p.add_argument("--config", help="YAML config path", default=None)
    args = p.parse_args()

    if args.config:
        cfg = load_config(args.config)
        a, b, max_rounds = build_agents(cfg)
    else:
        # Built-in default config
        issues = [Issue("price", 0.0, 1.0, 0.7), Issue("warranty", 0.0, 1.0, 0.3)]
        a = Agent("Alice", issues, 0.2, LinearConcessionPolicy(), ThresholdAcceptancePolicy(0.6))
        b = Agent("Bob", issues, 0.3, LinearConcessionPolicy(), ThresholdAcceptancePolicy(0.55))
        max_rounds = 7

    outcome = run_negotiation(a, b, max_rounds=max_rounds)

    print("agreement:", outcome.agreement)
    print("rounds:", outcome.rounds)
    print("utility_a:", round(outcome.utility_a, 4))
    print("utility_b:", round(outcome.utility_b, 4))
    if outcome.deal:
        print("deal:", {k: round(v, 4) for k, v in outcome.deal.values.items()})


if __name__ == "__main__":
    main()

