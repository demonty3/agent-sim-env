"""
Core data models for the negotiation simulator.
Defines entities, policies, and negotiation structures.
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np


# ===== ENUMS =====

class PolicyType(str, Enum):
    """Types of negotiation policies available."""
    LINEAR_CONCESSION = "linear_concession"
    TIT_FOR_TAT = "tit_for_tat"
    BOULWARE = "boulware"  # Hard early, concede late
    CONCEDER = "conceder"   # Concede early, hard late
    FIXED_THRESHOLD = "fixed_threshold"
    ADAPTIVE = "adaptive"


class OfferStatus(str, Enum):
    """Status of an offer in the negotiation."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"


# ===== CORE MODELS =====

class Issue(BaseModel):
    """A single negotiable issue (e.g., price, quantity, delivery date)."""
    name: str
    min_value: float
    max_value: float
    divisible: bool = True  # Can the issue be split?
    unit: Optional[str] = None  # e.g., "USD", "tons", "days"

    @validator('max_value')
    def validate_range(cls, v, values):
        if 'min_value' in values and v <= values['min_value']:
            raise ValueError('max_value must be greater than min_value')
        return v


class UtilityFunction(BaseModel):
    """Defines how an entity values different outcomes."""
    weights: Dict[str, float]  # Issue name -> importance weight
    ideal_values: Dict[str, float]  # Issue name -> ideal value
    reservation_values: Dict[str, float]  # Issue name -> walk-away point

    def calculate_utility(self, offer: Dict[str, float]) -> float:
        """Calculate utility score for a given offer."""
        total_utility = 0.0
        total_weight = sum(self.weights.values())

        for issue, value in offer.items():
            if issue not in self.weights:
                continue

            weight = self.weights[issue] / total_weight
            ideal = self.ideal_values.get(issue, value)
            reservation = self.reservation_values.get(issue, value)

            # Normalize to [0, 1] based on reservation and ideal
            if ideal != reservation:
                normalized = (value - reservation) / (ideal - reservation)
                normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
            else:
                normalized = 1.0 if value == ideal else 0.0

            total_utility += weight * normalized

        return total_utility


class PolicyParameters(BaseModel):
    """Parameters for negotiation policies."""
    accept_threshold: float = Field(0.7, ge=0, le=1)  # Min utility to accept
    initial_demand: float = Field(0.95, ge=0, le=1)  # Starting position
    concession_rate: float = Field(0.1, ge=0, le=1)  # How fast to concede
    patience: int = Field(10, ge=1)  # Rounds before major concession
    stubbornness: float = Field(0.5, ge=0, le=1)  # Resistance to change

    # Advanced parameters
    learning_rate: float = Field(0.1, ge=0, le=1)
    exploration_factor: float = Field(0.2, ge=0, le=1)

    class Config:
        extra = 'allow'  # Allow additional parameters for custom policies


class NegotiationPolicy(BaseModel):
    """Defines how an entity negotiates."""
    type: PolicyType
    params: PolicyParameters = Field(default_factory=PolicyParameters)

    def make_offer(self,
                   round_num: int,
                   history: List['Offer'],
                   utility_fn: UtilityFunction,
                   issues: List[Issue]) -> Dict[str, float]:
        """Generate an offer based on policy type and parameters."""

        if self.type == PolicyType.LINEAR_CONCESSION:
            return self._linear_concession_offer(round_num, utility_fn, issues)
        elif self.type == PolicyType.FIXED_THRESHOLD:
            return self._fixed_threshold_offer(utility_fn, issues)
        # Add other policy implementations...
        else:
            return self._default_offer(utility_fn, issues)

    def _linear_concession_offer(self, round_num: int,
                                  utility_fn: UtilityFunction,
                                  issues: List[Issue]) -> Dict[str, float]:
        """Linear concession from ideal to reservation over time."""
        concession_factor = min(1.0, round_num * self.params.concession_rate)
        offer = {}

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Linear interpolation
            value = ideal - concession_factor * (ideal - reservation)
            offer[issue.name] = value

        return offer

    def _fixed_threshold_offer(self, utility_fn: UtilityFunction,
                                issues: List[Issue]) -> Dict[str, float]:
        """Always offer at a fixed utility threshold."""
        # Start with ideal values
        offer = {issue.name: utility_fn.ideal_values.get(issue.name, issue.max_value)
                 for issue in issues}

        # Could add logic to find offer at exact threshold
        return offer

    def _default_offer(self, utility_fn: UtilityFunction,
                       issues: List[Issue]) -> Dict[str, float]:
        """Default offer at ideal values."""
        return {issue.name: utility_fn.ideal_values.get(issue.name,
                           (issue.max_value + issue.min_value) / 2)
                for issue in issues}


class Entity(BaseModel):
    """A negotiating party (country, company, etc.)."""
    name: str
    type: Literal["country", "company", "individual", "other"] = "country"
    utility_function: UtilityFunction
    policy: NegotiationPolicy

    # Constraints and limits
    max_rounds: int = Field(100, ge=1)
    min_acceptable_utility: float = Field(0.5, ge=0, le=1)

    # Optional attributes
    resources: Dict[str, float] = Field(default_factory=dict)
    relationships: Dict[str, float] = Field(default_factory=dict)  # Entity name -> relationship score

    def evaluate_offer(self, offer: Dict[str, float]) -> tuple[bool, float]:
        """Evaluate an offer and decide whether to accept."""
        utility = self.utility_function.calculate_utility(offer)
        accept = utility >= self.policy.params.accept_threshold
        return accept, utility


# ===== NEGOTIATION STRUCTURES =====

@dataclass
class Offer:
    """A single offer in the negotiation."""
    round_num: int
    proposer: str
    values: Dict[str, float]
    status: OfferStatus = OfferStatus.PENDING
    utility_scores: Dict[str, float] = field(default_factory=dict)  # Entity -> utility
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'round': self.round_num,
            'proposer': self.proposer,
            'values': self.values,
            'status': self.status.value,
            'utilities': self.utility_scores
        }


@dataclass
class NegotiationRound:
    """Complete information for one negotiation round."""
    round_num: int
    offers: List[Offer]
    active_proposer: str
    responses: Dict[str, bool] = field(default_factory=dict)  # Entity -> accept/reject

    def is_complete(self) -> bool:
        """Check if all parties have responded."""
        return len(self.responses) == len(self.offers[0].utility_scores)


class NegotiationOutcome(BaseModel):
    """Final outcome of a negotiation."""
    success: bool
    final_agreement: Optional[Dict[str, float]] = None
    rounds_taken: int
    final_utilities: Dict[str, float]
    transcript: List[NegotiationRound]

    # Analysis fields
    impasse_reason: Optional[str] = None
    pareto_optimal: Optional[bool] = None
    nash_bargaining_score: Optional[float] = None

    def summary(self) -> str:
        """Generate a text summary of the outcome."""
        if self.success:
            return f"✅ Agreement reached in {self.rounds_taken} rounds. " \
                   f"Average utility: {np.mean(list(self.final_utilities.values())):.2f}"
        else:
            return f"❌ Impasse after {self.rounds_taken} rounds. " \
                   f"Reason: {self.impasse_reason or 'Unknown'}"


# ===== CONFIGURATION =====

class SimulationConfig(BaseModel):
    """Configuration for a complete simulation run."""
    entities: List[Entity]
    issues: List[Issue]
    max_rounds: int = Field(100, ge=1)
    protocol: Literal["alternating", "simultaneous", "random"] = "alternating"

    # Advanced settings
    allow_coalition: bool = False
    allow_side_payments: bool = False
    information_type: Literal["complete", "incomplete"] = "complete"

    # Analysis settings
    track_pareto: bool = True
    calculate_nash: bool = True

    @validator('entities')
    def validate_entities(cls, v):
        if len(v) < 2:
            raise ValueError('Need at least 2 entities to negotiate')
        return v

    def to_yaml(self) -> str:
        """Export configuration to YAML format."""
        import yaml
        return yaml.dump(self.dict(), default_flow_style=False)
