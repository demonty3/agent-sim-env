"""
Negotiation protocol engine that orchestrates the negotiation process.
Supports simple alternating, simultaneous, and random proposer protocols.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from models import (
    Entity,
    Issue,
    Offer,
    OfferStatus,
    NegotiationRound,
    NegotiationOutcome,
    SimulationConfig,
)
from utilities import (
    calculate_nash_product,
    is_pareto_optimal,
)


class NegotiationEngine:
    """
    Minimal negotiation engine.

    - Alternating: entities take turns proposing one offer per round.
    - Simultaneous: all entities propose; pick the offer with max joint utility.
    - Random: randomly choose a proposer each round.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.entities: List[Entity] = config.entities
        self.issues: List[Issue] = config.issues
        self.max_rounds: int = config.max_rounds
        self.protocol: str = config.protocol
        self.current_round: int = 0

        # Transcript of rounds
        self.transcript: List[NegotiationRound] = []

    # ---------- Core Run ----------
    def run(self) -> NegotiationOutcome:
        if self.protocol == "alternating":
            return self._run_alternating()
        elif self.protocol == "simultaneous":
            return self._run_simultaneous()
        else:  # "random" or unknown -> treat as random proposer
            return self._run_random()

    # ---------- Protocol Implementations ----------
    def _run_alternating(self) -> NegotiationOutcome:
        proposer_index = 0
        last_utilities: Dict[str, float] = {e.name: 0.0 for e in self.entities}

        for r in range(1, self.max_rounds + 1):
            self.current_round = r
            proposer = self.entities[proposer_index]
            offer_values = proposer.policy.make_offer(
                r, self._flatten_history(), proposer.utility_function, self.issues
            )
            offer = self._evaluate_offer(offer_values, proposer.name)

            round_obj = NegotiationRound(
                round_num=r, offers=[offer], active_proposer=proposer.name
            )

            # Collect responses
            all_accept = True
            for entity in self.entities:
                accept, _ = entity.evaluate_offer(offer.values)
                round_obj.responses[entity.name] = accept
                if not accept:
                    all_accept = False

            # Update status and transcript
            offer.status = OfferStatus.ACCEPTED if all_accept else OfferStatus.COUNTERED
            self.transcript.append(round_obj)

            # Track last utilities
            last_utilities = offer.utility_scores.copy()

            if all_accept:
                return self._finalize_outcome(
                    success=True,
                    rounds_taken=r,
                    agreement=offer.values,
                    final_utilities=offer.utility_scores,
                )

            # Next proposer
            proposer_index = (proposer_index + 1) % len(self.entities)

        # Max rounds reached
        return self._finalize_outcome(
            success=False,
            rounds_taken=self.max_rounds,
            agreement=None,
            final_utilities=last_utilities,
            impasse_reason="max rounds reached",
        )

    def _run_simultaneous(self) -> NegotiationOutcome:
        last_utilities: Dict[str, float] = {e.name: 0.0 for e in self.entities}

        for r in range(1, self.max_rounds + 1):
            self.current_round = r
            offers: List[Offer] = []

            # Each entity proposes
            for entity in self.entities:
                values = entity.policy.make_offer(
                    r, self._flatten_history(), entity.utility_function, self.issues
                )
                offers.append(self._evaluate_offer(values, entity.name))

            # Choose the offer with max sum of utilities
            best_offer = max(
                offers,
                key=lambda o: float(sum(o.utility_scores.values())) if o.utility_scores else -1.0,
            )

            round_obj = NegotiationRound(
                round_num=r, offers=offers, active_proposer="simultaneous"
            )

            # Everyone evaluates the best offer
            all_accept = True
            for entity in self.entities:
                accept, _ = entity.evaluate_offer(best_offer.values)
                round_obj.responses[entity.name] = accept
                if not accept:
                    all_accept = False

            # Mark statuses (best marked accepted/countered)
            for off in offers:
                off.status = OfferStatus.ACCEPTED if off is best_offer and all_accept else OfferStatus.COUNTERED

            self.transcript.append(round_obj)
            last_utilities = best_offer.utility_scores.copy()

            if all_accept:
                return self._finalize_outcome(
                    success=True,
                    rounds_taken=r,
                    agreement=best_offer.values,
                    final_utilities=best_offer.utility_scores,
                )

        return self._finalize_outcome(
            success=False,
            rounds_taken=self.max_rounds,
            agreement=None,
            final_utilities=last_utilities,
            impasse_reason="max rounds reached",
        )

    def _run_random(self) -> NegotiationOutcome:
        last_utilities: Dict[str, float] = {e.name: 0.0 for e in self.entities}

        for r in range(1, self.max_rounds + 1):
            self.current_round = r
            proposer = random.choice(self.entities)
            values = proposer.policy.make_offer(
                r, self._flatten_history(), proposer.utility_function, self.issues
            )
            offer = self._evaluate_offer(values, proposer.name)

            round_obj = NegotiationRound(
                round_num=r, offers=[offer], active_proposer=proposer.name
            )

            all_accept = True
            for entity in self.entities:
                accept, _ = entity.evaluate_offer(offer.values)
                round_obj.responses[entity.name] = accept
                if not accept:
                    all_accept = False

            offer.status = OfferStatus.ACCEPTED if all_accept else OfferStatus.COUNTERED
            self.transcript.append(round_obj)
            last_utilities = offer.utility_scores.copy()

            if all_accept:
                return self._finalize_outcome(
                    success=True,
                    rounds_taken=r,
                    agreement=offer.values,
                    final_utilities=offer.utility_scores,
                )

        return self._finalize_outcome(
            success=False,
            rounds_taken=self.max_rounds,
            agreement=None,
            final_utilities=last_utilities,
            impasse_reason="max rounds reached",
        )

    # ---------- Helpers ----------
    def _evaluate_offer(self, values: Dict[str, float], proposer: str) -> Offer:
        offer = Offer(round_num=self.current_round, proposer=proposer, values=values)
        # Compute per-entity utilities
        utilities: Dict[str, float] = {}
        for entity in self.entities:
            utilities[entity.name] = entity.utility_function.calculate_utility(values)
        offer.utility_scores = utilities
        return offer

    def _flatten_history(self) -> List[Offer]:
        offers: List[Offer] = []
        for rnd in self.transcript:
            offers.extend(rnd.offers)
        return offers

    def _finalize_outcome(
        self,
        *,
        success: bool,
        rounds_taken: int,
        agreement: Optional[Dict[str, float]],
        final_utilities: Dict[str, float],
        impasse_reason: Optional[str] = None,
    ) -> NegotiationOutcome:
        # Optional analysis
        pareto_opt: Optional[bool] = None
        nash_score: Optional[float] = None
        if success and agreement is not None:
            if getattr(self.config, "track_pareto", False):
                try:
                    pareto_opt = is_pareto_optimal(agreement, self.entities, self.issues, samples=100)
                except Exception:
                    pareto_opt = None
            if getattr(self.config, "calculate_nash", False):
                try:
                    nash_score = calculate_nash_product(agreement, self.entities)
                except Exception:
                    nash_score = None

        outcome = NegotiationOutcome(
            success=success,
            final_agreement=agreement,
            rounds_taken=rounds_taken,
            final_utilities=final_utilities,
            transcript=self.transcript,
            impasse_reason=impasse_reason,
            pareto_optimal=pareto_opt,
            nash_bargaining_score=nash_score,
        )
        return outcome


class BatchNegotiationRunner:
    """Run many negotiations and analyze outcomes."""

    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config
        self.results: List[NegotiationOutcome] = []

    def run_batch(self, n_runs: int, vary_params: Optional[Dict] = None) -> List[NegotiationOutcome]:
        self.results = []
        for i in range(n_runs):
            config = self._apply_variations(self.base_config, vary_params, seed=i)
            engine = NegotiationEngine(config)
            self.results.append(engine.run())
        return self.results

    def analyze_results(self) -> Dict[str, float]:
        if not self.results:
            return {"success_rate": 0.0, "average_rounds": 0.0, "total_runs": 0}

        total_runs = len(self.results)
        successes = [r for r in self.results if r.success]
        success_rate = len(successes) / total_runs
        avg_rounds = float(np.mean([r.rounds_taken for r in self.results])) if self.results else 0.0

        # Average utilities across successful runs (simple average of means)
        avg_utils: Dict[str, float] = {}
        if successes:
            names = set().union(*[set(r.final_utilities.keys()) for r in successes])
            for name in names:
                vals = [r.final_utilities.get(name, 0.0) for r in successes]
                avg_utils[name] = float(np.mean(vals)) if vals else 0.0

        return {
            "success_rate": success_rate,
            "average_rounds": avg_rounds,
            "total_runs": total_runs,
            "average_utilities": avg_utils if avg_utils else None,
        }

    def _apply_variations(
        self, config: SimulationConfig, vary_params: Optional[Dict], seed: int
    ) -> SimulationConfig:
        if not vary_params:
            return config

        # Shallow copy with slight variation of thresholds/rates if requested
        import copy

        new_cfg = copy.deepcopy(config)
        rng = random.Random(seed)

        for entity in new_cfg.entities:
            params = entity.policy.params
            if vary_params.get("accept_threshold"):
                delta = rng.uniform(-0.05, 0.05)
                params.accept_threshold = float(np.clip(params.accept_threshold + delta, 0.0, 1.0))
            if vary_params.get("concession_rate"):
                delta = rng.uniform(-0.02, 0.02)
                params.concession_rate = float(np.clip(params.concession_rate + delta, 0.0, 1.0))

        return new_cfg
