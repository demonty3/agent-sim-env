from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .models import Outcome


def summarize_outcomes(outcomes: Iterable[Outcome]) -> dict:
    outs: List[Outcome] = list(outcomes)
    n = len(outs) or 1
    agreements = sum(1 for o in outs if o.agreement)
    avg_rounds = sum(o.rounds for o in outs) / n
    avg_u_a = sum(o.utility_a for o in outs) / n
    avg_u_b = sum(o.utility_b for o in outs) / n
    return {
        "count": len(outs),
        "agreement_rate": agreements / n,
        "avg_rounds": avg_rounds,
        "avg_utility_a": avg_u_a,
        "avg_utility_b": avg_u_b,
    }

