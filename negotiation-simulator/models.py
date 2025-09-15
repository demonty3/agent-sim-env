"""
Core data models for the negotiation simulator.
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from pydantic import model_validator  # type: ignore
except Exception:  # pragma: no cover
    model_validator = None  # type: ignore
try:
    from pydantic.v1 import validator
except Exception:  # pragma: no cover
    from pydantic import validator  # type: ignore
from enum import Enum
import numpy as np


class PolicyType(str, Enum):
    """Enumerate the supported negotiation policy strategies."""

    LINEAR_CONCESSION = "linear_concession"
    TIT_FOR_TAT = "tit_for_tat"
    BOULWARE = "boulware"
    CONCEDER = "conceder"
    FIXED_THRESHOLD = "fixed_threshold"
    ADAPTIVE = "adaptive"


class OfferStatus(str, Enum):
    """Possible lifecycle statuses for a proposed offer."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"


class Issue(BaseModel):
    """Describe a single negotiable issue and its feasible domain."""

    name: str
    min_value: float
    max_value: float
    divisible: bool = True
    unit: Optional[str] = None

    # Pydantic v1 compatibility validator
    @validator('max_value', allow_reuse=True)
    def _v1_validate_range(cls, v, values):  # type: ignore[override]
        if 'min_value' in values and v <= values['min_value']:
            raise ValueError('max_value must be greater than min_value')
        return v

    # Pydantic v2 validator to ensure cross-field validation works reliably
    if model_validator is not None:  # type: ignore[truthy-function]
        @model_validator(mode="after")  # type: ignore[misc]
        def _v2_validate_range(self):  # type: ignore[no-redef]
            if self.max_value <= self.min_value:
                raise ValueError('max_value must be greater than min_value')
            return self


class UtilityFunction(BaseModel):
    """Utility model capturing preferences for an entity across issues."""

    weights: Dict[str, float]
    ideal_values: Dict[str, float]
    reservation_values: Dict[str, float]

    def calculate_utility(self, offer: Dict[str, float]) -> float:
        """Compute the normalized utility of a candidate offer.

        Args:
            offer: Mapping of issue names to the proposed values.

        Returns:
            float: Weighted utility in the range ``[0.0, 1.0]`` where larger
            values are preferred.

        Side Effects:
            None.
        """

        total_utility = 0.0
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            return 0.0
        for issue, value in offer.items():
            if issue not in self.weights:
                continue
            weight = self.weights[issue] / total_weight
            ideal = self.ideal_values.get(issue, value)
            reservation = self.reservation_values.get(issue, value)
            if ideal != reservation:
                normalized = (value - reservation) / (ideal - reservation)
                normalized = max(0, min(1, normalized))
            else:
                normalized = 1.0 if value == ideal else 0.0
            total_utility += weight * normalized
        return total_utility


class PolicyParameters(BaseModel):
    """Configuration parameters that modulate a negotiation policy."""

    accept_threshold: float = Field(0.7, ge=0, le=1)
    initial_demand: float = Field(0.95, ge=0, le=1)
    concession_rate: float = Field(0.1, ge=0, le=1)
    patience: int = Field(10, ge=1)
    stubbornness: float = Field(0.5, ge=0, le=1)
    learning_rate: float = Field(0.1, ge=0, le=1)
    exploration_factor: float = Field(0.2, ge=0, le=1)

    class Config:
        extra = 'allow'


class NegotiationPolicy(BaseModel):
    """Select and execute the strategy used to construct counter-offers."""

    type: PolicyType
    params: PolicyParameters = Field(default_factory=PolicyParameters)

    def make_offer(self, round_num: int, history: List['Offer'], utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        """Generate an offer according to the configured policy type.

        Args:
            round_num: One-based round counter used for time-dependent
                strategies.
            history: Chronological list of prior :class:`Offer` instances.
            utility_fn: Utility function belonging to the proposing entity.
            issues: Negotiable issues that must appear in the returned offer.

        Returns:
            Dict[str, float]: Mapping from issue names to proposed values that
            complies with the chosen strategy and parameterization.

        Side Effects:
            May temporarily adjust internal parameters or draw random samples
            for adaptive behaviors but restores the original values before
            returning.
        """

        if self.type == PolicyType.LINEAR_CONCESSION:
            return self._linear_concession_offer(round_num, utility_fn, issues)
        elif self.type == PolicyType.FIXED_THRESHOLD:
            return self._fixed_threshold_offer(utility_fn, issues)
        elif self.type == PolicyType.TIT_FOR_TAT:
            return self._tit_for_tat_offer(round_num, history, utility_fn, issues)
        elif self.type == PolicyType.BOULWARE:
            return self._boulware_offer(round_num, utility_fn, issues)
        elif self.type == PolicyType.CONCEDER:
            return self._conceder_offer(round_num, utility_fn, issues)
        elif self.type == PolicyType.ADAPTIVE:
            return self._adaptive_offer(round_num, history, utility_fn, issues)
        else:
            return self._default_offer(utility_fn, issues)

    def _linear_concession_offer(self, round_num: int, utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        concession_factor = min(1.0, round_num * self.params.concession_rate)
        offer = {}
        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)
            value = ideal - concession_factor * (ideal - reservation)
            offer[issue.name] = value
        return offer

    def _fixed_threshold_offer(self, utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        return {issue.name: utility_fn.ideal_values.get(issue.name, issue.max_value) for issue in issues}

    def _default_offer(self, utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        return {issue.name: utility_fn.ideal_values.get(issue.name, (issue.max_value + issue.min_value) / 2) for issue in issues}

    # --- Additional strategy helpers ---
    def _tit_for_tat_offer(self, round_num: int, history: List['Offer'], utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        # If insufficient history, behave like linear
        if len(history) < 2:
            return self._linear_concession_offer(round_num, utility_fn, issues)

        # Estimate recent concession magnitude across issues from last two offers
        prev, last = history[-2], history[-1]
        concession_mag = 0.0
        count = 0
        for issue in issues:
            if issue.name in prev.values and issue.name in last.values:
                concession_mag += abs(last.values[issue.name] - prev.values[issue.name])
                count += 1
        avg_concession = (concession_mag / max(count, 1)) if count else 0.0

        # Map average concession to a factor relative to the issue ranges
        # Normalize by average issue range
        avg_range = np.mean([abs(i.max_value - i.min_value) for i in issues]) if issues else 1.0
        mirror_factor = np.clip(avg_concession / max(avg_range, 1e-9), 0.0, 1.0)

        # Apply mirrored concession tempered by stubbornness
        effective_rate = np.clip(self.params.concession_rate + mirror_factor * (1.0 - self.params.stubbornness), 0.0, 1.0)
        # Use linear path with adjusted rate
        saved = self.params.concession_rate
        try:
            self.params.concession_rate = effective_rate
            return self._linear_concession_offer(round_num, utility_fn, issues)
        finally:
            self.params.concession_rate = saved

    def _boulware_offer(self, round_num: int, utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        # Slow early concessions: use a convex (power > 1) schedule based on rounds
        base = round_num * max(self.params.concession_rate, 1e-6)
        concession_factor = min(1.0, base ** 2)
        offer = {}
        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)
            value = ideal - concession_factor * (ideal - reservation)
            offer[issue.name] = value
        return offer

    def _conceder_offer(self, round_num: int, utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        # Fast concessions: amplify rate
        concession_factor = min(1.0, round_num * self.params.concession_rate * 2.0)
        offer = {}
        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)
            value = ideal - concession_factor * (ideal - reservation)
            offer[issue.name] = value
        return offer

    def _adaptive_offer(self, round_num: int, history: List['Offer'], utility_fn: UtilityFunction, issues: List[Issue]) -> Dict[str, float]:
        # Adjust rate based on recent joint utility trend and add small exploration noise
        recent = history[-5:]
        joint_utils = []
        for o in recent:
            if o.utility_scores:
                joint_utils.append(float(np.sum(list(o.utility_scores.values()))))

        rate = self.params.concession_rate
        if len(joint_utils) >= 2:
            delta = joint_utils[-1] - joint_utils[0]
            # If joint utility improving, keep steady; if not, increase rate by learning_rate
            if delta <= 0:
                rate = np.clip(rate + self.params.learning_rate * 0.2, 0.0, 1.0)
            else:
                rate = np.clip(rate * (1.0 - self.params.learning_rate * 0.1), 0.0, 1.0)

        # Temporarily apply adjusted rate and add exploration noise per issue
        saved = self.params.concession_rate
        try:
            self.params.concession_rate = rate
            base_offer = self._linear_concession_offer(round_num, utility_fn, issues)
        finally:
            self.params.concession_rate = saved

        # Exploration noise
        noise_scale = self.params.exploration_factor * 0.05
        for issue in issues:
            rng = (issue.max_value - issue.min_value)
            if rng > 0 and issue.name in base_offer:
                base_offer[issue.name] = float(np.clip(
                    base_offer[issue.name] + np.random.normal(0, rng * noise_scale),
                    issue.min_value,
                    issue.max_value
                ))
        return base_offer


class Entity(BaseModel):
    """Participating negotiator with preferences, policy, and resources."""

    name: str
    type: Literal["country", "company", "individual", "other"] = "country"
    utility_function: UtilityFunction
    policy: NegotiationPolicy
    max_rounds: int = Field(100, ge=1)
    # Default minimum acceptable utility for ZOPA checks
    # Note: lower default improves ZOPA sampling in generic scenarios/tests.
    min_acceptable_utility: float = Field(0.1, ge=0, le=1)
    resources: Dict[str, float] = Field(default_factory=dict)
    relationships: Dict[str, float] = Field(default_factory=dict)

    def evaluate_offer(self, offer: Dict[str, float]) -> tuple[bool, float]:
        """Assess whether the entity accepts the given offer.

        Args:
            offer: Proposed issue values to evaluate.

        Returns:
            tuple[bool, float]: Boolean acceptance decision and the computed
            utility score for the offer.

        Side Effects:
            None.
        """

        utility = self.utility_function.calculate_utility(offer)
        accept = utility >= self.policy.params.accept_threshold
        return accept, utility


@dataclass
class Offer:
    """Immutable representation of a single offer exchanged in negotiation."""

    round_num: int
    proposer: str
    values: Dict[str, float]
    status: OfferStatus = OfferStatus.PENDING
    utility_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert the offer to a serializable mapping.

        Returns:
            dict: Dictionary representation of the offer values and metadata.

        Side Effects:
            None.
        """

        return {
            'round': self.round_num,
            'proposer': self.proposer,
            'values': self.values,
            'status': self.status.value,
            'utilities': self.utility_scores
        }


@dataclass
class NegotiationRound:
    """Aggregate offers and responses recorded for a negotiation round."""

    round_num: int
    offers: List[Offer]
    active_proposer: str
    responses: Dict[str, bool] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Determine whether every entity responded during this round.

        Returns:
            bool: ``True`` if the number of responses equals the number of
            tracked entities; ``False`` otherwise.

        Side Effects:
            None.
        """

        return len(self.responses) == len(self.offers[0].utility_scores)


class NegotiationOutcome(BaseModel):
    """Summary of the negotiation process and its resulting agreement."""

    success: bool
    final_agreement: Optional[Dict[str, float]] = None
    rounds_taken: int
    final_utilities: Dict[str, float]
    transcript: List[NegotiationRound]
    impasse_reason: Optional[str] = None
    pareto_optimal: Optional[bool] = None
    nash_bargaining_score: Optional[float] = None

    def summary(self) -> str:
        """Provide a human-readable synopsis of the negotiation result.

        Returns:
            str: Emoji-prefixed sentence that captures agreement status and
            supporting metrics.

        Side Effects:
            None.
        """

        if self.success:
            return f"✅ Agreement reached in {self.rounds_taken} rounds. " \
                   f"Average utility: {np.mean(list(self.final_utilities.values())):.2f}"
        else:
            return f"❌ Impasse after {self.rounds_taken} rounds. " \
                   f"Reason: {self.impasse_reason or 'Unknown'}"


class SimulationConfig(BaseModel):
    """Container for simulation participants, issues, and protocol options."""

    entities: List[Entity]
    issues: List[Issue]
    max_rounds: int = Field(100, ge=1)
    protocol: Literal["alternating", "simultaneous", "random"] = "alternating"
    allow_coalition: bool = False
    allow_side_payments: bool = False
    information_type: Literal["complete", "incomplete"] = "complete"
    track_pareto: bool = True
    calculate_nash: bool = True

    @validator('entities', allow_reuse=True)
    def validate_entities(cls, v):
        """Ensure there are enough participants for a negotiation session.

        Args:
            v: Collection of configured entities.

        Returns:
            List[Entity]: The validated list of participants.

        Side Effects:
            Raises a :class:`ValueError` if fewer than two entities are
            provided.
        """

        if len(v) < 2:
            raise ValueError('Need at least 2 entities to negotiate')
        return v

    def to_yaml(self) -> str:
        """Serialize the configuration to YAML for persistence or sharing.

        Returns:
            str: YAML document describing the simulation parameters.

        Side Effects:
            Imports :mod:`yaml` to perform the serialization.
        """

        import yaml
        return yaml.dump(self.dict(), default_flow_style=False)
