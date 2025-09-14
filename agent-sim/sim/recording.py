from __future__ import annotations

from typing import Dict, List, Optional

from .models import Deal


class Recorder:
    def __init__(self) -> None:
        self._events: List[Dict[str, object]] = []

    def record_offer(self, round_index: int, proposer: str, offer: Deal) -> None:
        self._events.append(
            {
                "type": "offer",
                "round": round_index,
                "proposer": proposer,
                "offer": offer.values,
            }
        )

    def record_acceptance(self, round_index: int, responder: str) -> None:
        self._events.append(
            {
                "type": "accept",
                "round": round_index,
                "responder": responder,
            }
        )

    def record_end(self, reason: str) -> None:
        self._events.append({"type": "end", "reason": reason})

    def transcript(self) -> List[Dict[str, object]]:
        return list(self._events)

