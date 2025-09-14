from __future__ import annotations

from sim.models import Agent, Issue
from sim.policies import LinearConcessionPolicy, ThresholdAcceptancePolicy
from sim.protocol import run_negotiation


def test_protocol_reaches_agreement_with_reasonable_thresholds():
    issues = [Issue("x", 0.0, 1.0, 1.0)]
    a = Agent("A", issues, 0.2, LinearConcessionPolicy(), ThresholdAcceptancePolicy(0.5))
    b = Agent("B", issues, 0.2, LinearConcessionPolicy(), ThresholdAcceptancePolicy(0.5))
    outcome = run_negotiation(a, b, max_rounds=5)
    assert outcome.agreement is True
    assert outcome.deal is not None
    assert outcome.rounds <= 5

