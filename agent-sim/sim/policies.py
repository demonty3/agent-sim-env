from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import Deal, Issue
from .utilities import normalize_weights


class ConcessionPolicy(ABC):
    @abstractmethod
    def next_offer(
        self,
        round_index: int,
        max_rounds: int,
        last_offer: Optional[Deal],
        agent: "Agent",  # type: ignore[name-defined]
    ) -> Deal: ...


class LinearConcessionPolicy(ConcessionPolicy):
    """Linearly lowers aspiration from 1.0 to reservation across rounds."""

    def next_offer(
        self,
        round_index: int,
        max_rounds: int,
        last_offer: Optional[Deal],
        agent: "Agent",  # type: ignore[name-defined]
    ) -> Deal:
        t = min(1.0, round_index / max(1, max_rounds - 1))
        # aspiration goes from 1 -> reservation
        aspiration = 1.0 - t * (1.0 - agent.reservation_value)
        issues = normalize_weights(agent.issues)
        # Construct a deal that hits the aspiration as a weighted sum.
        # Here we simply target the same aspiration on each issue dimension.
        values = {}
        for i in issues:
            values[i.name] = i.min_value + aspiration * (i.max_value - i.min_value)
        return Deal(values)


class AcceptancePolicy(ABC):
    @abstractmethod
    def accept(
        self,
        round_index: int,
        max_rounds: int,
        offer: Deal,
        agent: "Agent",  # type: ignore[name-defined]
    ) -> bool: ...


class ThresholdAcceptancePolicy(AcceptancePolicy):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def accept(
        self,
        round_index: int,
        max_rounds: int,
        offer: Deal,
        agent: "Agent",  # type: ignore[name-defined]
    ) -> bool:
        # Be slightly more lenient in the final rounds
        if round_index >= max_rounds - 1:
            return True
        return agent.utility(offer) >= self.threshold

