from __future__ import annotations

from sim.models import Agent, Deal, Issue
from sim.policies import ThresholdAcceptancePolicy


def test_threshold_acceptance_policy():
    issues = [Issue("x", 0.0, 1.0, 1.0)]
    agent = Agent("A", issues, reservation_value=0.0, acceptance_policy=ThresholdAcceptancePolicy(0.6))
    assert agent.accept(0, 10, Deal({"x": 0.7})) is True
    assert agent.accept(0, 10, Deal({"x": 0.4})) is False

