"""Shared pytest fixtures for negotiation simulator tests."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models import (
    Entity,
    Issue,
    NegotiationPolicy,
    PolicyParameters,
    PolicyType,
    SimulationConfig,
    UtilityFunction,
)


@pytest.fixture
def simple_issues():
    """Create simple test issues."""
    return [
        Issue(name="price", min_value=100, max_value=200, unit="USD"),
        Issue(name="quantity", min_value=10, max_value=100, unit="units"),
    ]


@pytest.fixture
def simple_entities(simple_issues):
    """Create simple test entities."""
    buyer = Entity(
        name="Buyer",
        utility_function=UtilityFunction(
            weights={"price": 0.7, "quantity": 0.3},
            ideal_values={"price": 100, "quantity": 100},
            reservation_values={"price": 180, "quantity": 20},
        ),
        policy=NegotiationPolicy(
            type=PolicyType.LINEAR_CONCESSION,
            params=PolicyParameters(accept_threshold=0.6, concession_rate=0.1),
        ),
    )

    seller = Entity(
        name="Seller",
        utility_function=UtilityFunction(
            weights={"price": 0.8, "quantity": 0.2},
            ideal_values={"price": 200, "quantity": 10},
            reservation_values={"price": 120, "quantity": 80},
        ),
        policy=NegotiationPolicy(
            type=PolicyType.TIT_FOR_TAT,
            params=PolicyParameters(accept_threshold=0.6),
        ),
    )

    return [buyer, seller]


@pytest.fixture
def simple_config(simple_entities, simple_issues):
    """Create simple test configuration."""
    return SimulationConfig(
        entities=simple_entities,
        issues=simple_issues,
        max_rounds=50,
        protocol="alternating",
    )
