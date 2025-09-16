"""Performance-oriented tests for batch negotiation workflows."""

import sys
import time

import numpy as np
import pytest

from models import (
    Entity,
    Issue,
    NegotiationPolicy,
    PolicyType,
    SimulationConfig,
    UtilityFunction,
)
from protocol import BatchNegotiationRunner, NegotiationEngine


@pytest.mark.slow
def test_large_batch_performance(simple_config):
    """Ensure batch simulations complete within resource expectations."""
    start_time = time.time()
    runner = BatchNegotiationRunner(simple_config)
    results = runner.run_batch(100)
    elapsed = time.time() - start_time

    assert len(results) == 100
    assert elapsed < 30  # Should complete 100 simulations in under 30 seconds

    total_size = sum(sys.getsizeof(r) for r in results)
    assert total_size < 10 * 1024 * 1024  # Less than 10MB for results


@pytest.mark.slow
def test_complex_scenario_performance():
    """Stress test the negotiation engine with many issues and entities."""
    np.random.seed(42)

    issues = [
        Issue(name=f"issue_{i}", min_value=0, max_value=100)
        for i in range(10)
    ]

    entities = []
    for i in range(5):
        weights = {f"issue_{j}": np.random.random() for j in range(10)}
        ideal = {f"issue_{j}": np.random.uniform(20, 80) for j in range(10)}
        reservation = {f"issue_{j}": np.random.uniform(10, 90) for j in range(10)}

        entities.append(
            Entity(
                name=f"Entity_{i}",
                utility_function=UtilityFunction(
                    weights=weights,
                    ideal_values=ideal,
                    reservation_values=reservation,
                ),
                policy=NegotiationPolicy(type=PolicyType.LINEAR_CONCESSION),
            )
        )

    config = SimulationConfig(
        entities=entities,
        issues=issues,
        max_rounds=50,
        protocol="simultaneous",
    )

    start_time = time.time()
    engine = NegotiationEngine(config)
    outcome = engine.run()
    elapsed = time.time() - start_time

    assert outcome is not None
    assert elapsed < 5  # Complex negotiation should complete in under 5 seconds
