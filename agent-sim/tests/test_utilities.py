from __future__ import annotations

from sim.models import Deal, Issue
from sim.utilities import compute_utility, normalize_weights


def test_normalize_weights_sums_to_one():
    issues = [Issue("a", 0, 1, 2.0), Issue("b", 0, 1, 1.0)]
    norm = normalize_weights(issues)
    total = sum(i.weight for i in norm)
    assert abs(total - 1.0) < 1e-9


def test_compute_utility_monotonic_in_issue_values():
    issues = [Issue("x", 0, 1, 1.0)]
    low = Deal({"x": 0.25})
    high = Deal({"x": 0.75})
    assert compute_utility(issues, high) > compute_utility(issues, low)

