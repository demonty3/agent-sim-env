"""
LLM Advisory module for intelligent parameter tuning and strategy recommendations.
This module analyzes negotiation outcomes and suggests parameter adjustments.
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from models import (
    Entity, Issue, NegotiationOutcome, SimulationConfig,
    PolicyType, PolicyParameters
)
from protocol import BatchNegotiationRunner
from utilities import calculate_concession_rate, analyze_negotiation_space


# ===== ADVISORY TYPES =====

class AdviceType(str, Enum):
    """Types of advice the advisor can provide."""
    PARAMETER_TUNING = "parameter_tuning"
    STRATEGY_CHANGE = "strategy_change"
    ISSUE_REFRAMING = "issue_reframing"
    COALITION_SUGGESTION = "coalition_suggestion"
    EXPLANATION = "explanation"


@dataclass
class ParameterSuggestion:
    """A suggested parameter change."""
    entity_name: str
    parameter_path: str
    current_value: float
    suggested_value: float
    rationale: str
    confidence: float  # 0-1 confidence in the suggestion


@dataclass
class StrategySuggestion:
    """A suggested strategy change."""
    entity_name: str
    current_policy: PolicyType
    suggested_policy: PolicyType
    rationale: str
    expected_improvement: float


@dataclass
class AdvisoryReport:
    """Complete advisory report with all suggestions."""
    outcome_analysis: str
    parameter_suggestions: List[ParameterSuggestion]
    strategy_suggestions: List[StrategySuggestion]
    key_insights: List[str]
    success_probability: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'outcome_analysis': self.outcome_analysis,
            'parameter_suggestions': [
                {
                    'entity': s.entity_name,
                    'parameter': s.parameter_path,
                    'current': s.current_value,
                    'suggested': s.suggested_value,
                    'rationale': s.rationale,
                    'confidence': s.confidence
                }
                for s in self.parameter_suggestions
            ],
            'strategy_suggestions': [
                {
                    'entity': s.entity_name,
                    'current_policy': s.current_policy.value,
                    'suggested_policy': s.suggested_policy.value,
                    'rationale': s.rationale,
                    'expected_improvement': s.expected_improvement
                }
                for s in self.strategy_suggestions
            ],
            'key_insights': self.key_insights,
            'success_probability': self.success_probability
        }


# ===== NEGOTIATION ADVISOR =====

class NegotiationAdvisor:
    """
    Analyzes negotiation outcomes and provides strategic advice.
    In V2, this will integrate with LLM APIs for more sophisticated analysis.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.learning_history: List[Tuple[SimulationConfig, NegotiationOutcome]] = []

    def analyze_outcome(self,
                        config: SimulationConfig,
                        outcome: NegotiationOutcome,
                        historical_outcomes: Optional[List[NegotiationOutcome]] = None) -> AdvisoryReport:
        """
        Analyze a negotiation outcome and provide recommendations.
        """
        # Store for learning
        self.learning_history.append((config, outcome))

        # Analyze the outcome
        if outcome.success:
            analysis = self._analyze_successful_negotiation(config, outcome)
        else:
            analysis = self._analyze_failed_negotiation(config, outcome)

        # Generate suggestions
        param_suggestions = self._generate_parameter_suggestions(config, outcome, historical_outcomes)
        strategy_suggestions = self._generate_strategy_suggestions(config, outcome)
        insights = self._extract_key_insights(config, outcome)

        # Estimate success probability with current settings
        success_prob = self._estimate_success_probability(config, historical_outcomes)

        return AdvisoryReport(
            outcome_analysis=analysis,
            parameter_suggestions=param_suggestions,
            strategy_suggestions=strategy_suggestions,
            key_insights=insights,
            success_probability=success_prob
        )

    def _analyze_successful_negotiation(self,
                                       config: SimulationConfig,
                                       outcome: NegotiationOutcome) -> str:
        """Analyze why a negotiation succeeded."""

        # Calculate metrics
        avg_utility = sum(outcome.final_utilities.values()) / len(outcome.final_utilities)
        utility_variance = self._calculate_variance(list(outcome.final_utilities.values()))

        # Build analysis
        analysis_parts = []

        if avg_utility > 0.75:
            analysis_parts.append("Excellent outcome with high satisfaction for all parties.")
        elif avg_utility > 0.6:
            analysis_parts.append("Good outcome with reasonable satisfaction levels.")
        else:
            analysis_parts.append("Agreement reached but with relatively low satisfaction.")

        if utility_variance < 0.05:
            analysis_parts.append("Very balanced deal with similar utility for all parties.")
        elif utility_variance > 0.15:
            analysis_parts.append("Imbalanced outcome - some parties gained much more than others.")

        if outcome.rounds_taken < config.max_rounds * 0.3:
            analysis_parts.append("Quick agreement suggests well-aligned interests.")
        elif outcome.rounds_taken > config.max_rounds * 0.8:
            analysis_parts.append("Prolonged negotiation indicates difficult alignment.")

        if outcome.pareto_optimal:
            analysis_parts.append("The outcome is Pareto optimal - no improvements possible without harming someone.")
        elif outcome.pareto_optimal is False:
            analysis_parts.append("Sub-optimal outcome - mutual improvements are possible.")

        return " ".join(analysis_parts)

    def _analyze_failed_negotiation(self,
                                   config: SimulationConfig,
                                   outcome: NegotiationOutcome) -> str:
        """Analyze why a negotiation failed."""

        analysis_parts = [f"Negotiation failed: {outcome.impasse_reason}."]

        # Analyze based on impasse reason
        if "No ZOPA" in outcome.impasse_reason:
            analysis_parts.append(
                "Fundamental incompatibility between parties' requirements. "
                "Consider adjusting reservation values or ideal positions."
            )
        elif "deadlock" in outcome.impasse_reason.lower():
            analysis_parts.append(
                "Parties got stuck in repetitive patterns. "
                "Consider increasing concession rates or reducing stubbornness."
            )
        elif "max rounds" in outcome.impasse_reason.lower():
            analysis_parts.append(
                "Parties were making progress but too slowly. "
                "Consider increasing concession rates or patience parameters."
            )
        elif "unlikely to converge" in outcome.impasse_reason.lower():
            analysis_parts.append(
                "Divergent negotiation trajectories detected. "
                "Policies may be incompatible - consider switching strategies."
            )

        # Analyze final positions if available
        if outcome.final_utilities:
            min_utility = min(outcome.final_utilities.values())
            max_utility = max(outcome.final_utilities.values())

            if max_utility < 0.5:
                analysis_parts.append("All parties had low utility - significant gap from acceptable outcomes.")
            elif min_utility < 0.3:
                analysis_parts.append("At least one party was far from acceptable terms.")

        return " ".join(analysis_parts)

    def _generate_parameter_suggestions(self,
                                       config: SimulationConfig,
                                       outcome: NegotiationOutcome,
                                       historical: Optional[List[NegotiationOutcome]]) -> List[ParameterSuggestion]:
        """Generate specific parameter adjustment suggestions."""

        suggestions = []

        # Analyze each entity's performance
        for entity in config.entities:
            entity_utility = outcome.final_utilities.get(entity.name, 0)

            # If utility is too low, suggest adjustments
            if not outcome.success or entity_utility < entity.min_acceptable_utility:

                # Suggest lower acceptance threshold if too rigid
                if entity.policy.params.accept_threshold > 0.7:
                    suggestions.append(ParameterSuggestion(
                        entity_name=entity.name,
                        parameter_path="accept_threshold",
                        current_value=entity.policy.params.accept_threshold,
                        suggested_value=max(0.6, entity.policy.params.accept_threshold - 0.1),
                        rationale=f"{entity.name} may be too rigid. Lower acceptance threshold could enable deals.",
                        confidence=0.7
                    ))

                # Suggest higher concession rate if too stubborn
                if entity.policy.params.concession_rate < 0.1:
                    suggestions.append(ParameterSuggestion(
                        entity_name=entity.name,
                        parameter_path="concession_rate",
                        current_value=entity.policy.params.concession_rate,
                        suggested_value=min(0.2, entity.policy.params.concession_rate + 0.05),
                        rationale=f"{entity.name} concedes too slowly. Faster concession could reach agreement.",
                        confidence=0.6
                    ))

                # Analyze concession patterns from history
                if outcome.transcript:
                    flat_history = []
                    for round_obj in outcome.transcript:
                        flat_history.extend(round_obj.offers)

                    concession_rate = calculate_concession_rate(flat_history, entity.name)

                    if concession_rate < 0.01:  # Almost no concession
                        suggestions.append(ParameterSuggestion(
                            entity_name=entity.name,
                            parameter_path="stubbornness",
                            current_value=entity.policy.params.stubbornness,
                            suggested_value=max(0.2, entity.policy.params.stubbornness - 0.2),
                            rationale=f"{entity.name} shows no flexibility. Reducing stubbornness is critical.",
                            confidence=0.85
                        ))

            # If doing well, suggest minor optimizations
            elif outcome.success and entity_utility > 0.8:
                if entity.policy.params.initial_demand < 0.95:
                    suggestions.append(ParameterSuggestion(
                        entity_name=entity.name,
                        parameter_path="initial_demand",
                        current_value=entity.policy.params.initial_demand,
                        suggested_value=min(0.98, entity.policy.params.initial_demand + 0.05),
                        rationale=f"{entity.name} has room to be more ambitious initially.",
                        confidence=0.5
                    ))

        return suggestions

    def _generate_strategy_suggestions(self,
                                      config: SimulationConfig,
                                      outcome: NegotiationOutcome) -> List[StrategySuggestion]:
        """Suggest policy/strategy changes."""

        suggestions = []

        for entity in config.entities:
            current_policy = entity.policy.type
            entity_utility = outcome.final_utilities.get(entity.name, 0)

            # Failed negotiations - suggest strategy switch
            if not outcome.success:

                if current_policy == PolicyType.BOULWARE:
                    # Boulware failed - try something softer
                    suggestions.append(StrategySuggestion(
                        entity_name=entity.name,
                        current_policy=current_policy,
                        suggested_policy=PolicyType.LINEAR_CONCESSION,
                        rationale="Boulware strategy too rigid. Linear concession offers more flexibility.",
                        expected_improvement=0.3
                    ))

                elif current_policy == PolicyType.TIT_FOR_TAT:
                    # Tit-for-tat may create deadlock
                    suggestions.append(StrategySuggestion(
                        entity_name=entity.name,
                        current_policy=current_policy,
                        suggested_policy=PolicyType.ADAPTIVE,
                        rationale="Tit-for-tat may be creating deadlock. Adaptive strategy breaks patterns.",
                        expected_improvement=0.25
                    ))

                elif current_policy == PolicyType.FIXED_THRESHOLD:
                    # Fixed threshold too inflexible
                    suggestions.append(StrategySuggestion(
                        entity_name=entity.name,
                        current_policy=current_policy,
                        suggested_policy=PolicyType.CONCEDER,
                        rationale="Fixed threshold lacks flexibility. Conceder strategy enables movement.",
                        expected_improvement=0.2
                    ))

            # Successful but suboptimal
            elif outcome.success and entity_utility < 0.6:

                if current_policy == PolicyType.CONCEDER:
                    suggestions.append(StrategySuggestion(
                        entity_name=entity.name,
                        current_policy=current_policy,
                        suggested_policy=PolicyType.TIT_FOR_TAT,
                        rationale="Conceding too much. Tit-for-tat ensures reciprocity.",
                        expected_improvement=0.15
                    ))

                elif current_policy == PolicyType.LINEAR_CONCESSION:
                    suggestions.append(StrategySuggestion(
                        entity_name=entity.name,
                        current_policy=current_policy,
                        suggested_policy=PolicyType.BOULWARE,
                        rationale="Too predictable. Boulware maintains stronger position longer.",
                        expected_improvement=0.1
                    ))

        return suggestions

    def _extract_key_insights(self,
                             config: SimulationConfig,
                             outcome: NegotiationOutcome) -> List[str]:
        """Extract key insights from the negotiation."""

        insights = []

        # Analyze power dynamics
        if outcome.final_utilities:
            utilities = outcome.final_utilities
            max_entity = max(utilities, key=utilities.get)
            min_entity = min(utilities, key=utilities.get)

            if utilities[max_entity] - utilities[min_entity] > 0.3:
                insights.append(
                    f"{max_entity} dominated the negotiation, gaining {utilities[max_entity]:.2f} utility "
                    f"vs {min_entity}'s {utilities[min_entity]:.2f}"
                )

        # Identify bottlenecks
        if outcome.transcript and len(outcome.transcript) > 10:
            # Check for repeated rejections
            rejection_count = {}
            for round_obj in outcome.transcript:
                for entity_name, accepted in round_obj.responses.items():
                    if not accepted:
                        rejection_count[entity_name] = rejection_count.get(entity_name, 0) + 1

            if rejection_count:
                blocker = max(rejection_count, key=rejection_count.get)
                if rejection_count[blocker] > len(outcome.transcript) * 0.5:
                    insights.append(f"{blocker} was the primary blocker, rejecting {rejection_count[blocker]} offers")

        # Protocol effectiveness
        if config.protocol == "simultaneous" and outcome.success:
            insights.append("Simultaneous protocol worked well for this multi-party scenario")
        elif config.protocol == "alternating" and not outcome.success:
            insights.append("Consider simultaneous protocol for better multi-party coordination")

        # Issue coupling
        if len(config.issues) > 2:
            insights.append(f"Complex {len(config.issues)}-issue negotiation may benefit from issue bundling")

        # Zone analysis
        space_analysis = analyze_negotiation_space(
            config.entities, config.issues, samples=100, rng=config.create_rng()
        )
        if space_analysis['has_zopa']:
            zopa_size = space_analysis['zopa_size']
            if zopa_size < 10:
                insights.append(f"Very small ZOPA ({zopa_size} solutions) makes agreement difficult")
            elif zopa_size > 50:
                insights.append(f"Large ZOPA ({zopa_size} solutions) - agreement should be achievable")

        return insights

    def _estimate_success_probability(self,
                                     config: SimulationConfig,
                                     historical: Optional[List[NegotiationOutcome]]) -> float:
        """Estimate probability of success with current configuration."""

        # Quick space analysis
        space_analysis = analyze_negotiation_space(
            config.entities, config.issues, samples=200, rng=config.create_rng()
        )

        if not space_analysis['has_zopa']:
            return 0.0

        # Base probability from ZOPA size
        zopa_size = space_analysis['zopa_size']
        base_prob = min(0.9, zopa_size / 100)  # More ZOPA = higher chance

        # Adjust based on policy compatibility
        policy_adjustment = 0.0
        policies = [e.policy.type for e in config.entities]

        # Good combinations
        if PolicyType.CONCEDER in policies and PolicyType.LINEAR_CONCESSION in policies:
            policy_adjustment += 0.1
        if PolicyType.ADAPTIVE in policies:
            policy_adjustment += 0.05

        # Bad combinations
        if policies.count(PolicyType.BOULWARE) > 1:
            policy_adjustment -= 0.2  # Multiple hard negotiators
        if policies.count(PolicyType.TIT_FOR_TAT) > 1:
            policy_adjustment -= 0.1  # Can create deadlocks

        # Adjust based on acceptance thresholds
        avg_threshold = sum(e.policy.params.accept_threshold for e in config.entities) / len(config.entities)
        if avg_threshold > 0.75:
            policy_adjustment -= 0.15  # Too demanding
        elif avg_threshold < 0.6:
            policy_adjustment += 0.1  # More flexible

        # Historical performance if available
        if historical and len(historical) >= 10:
            recent_success = sum(1 for o in historical[-10:] if o.success) / 10
            historical_weight = 0.3
            base_prob = base_prob * (1 - historical_weight) + recent_success * historical_weight

        return max(0.0, min(1.0, base_prob + policy_adjustment))

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def suggest_initial_parameters(self,
                                  entities: List[Entity],
                                  issues: List[Issue]) -> Dict[str, PolicyParameters]:
        """
        Suggest initial parameters based on entity characteristics.
        This is useful for bootstrapping new negotiations.
        """
        suggestions = {}

        # Analyze the negotiation space
        temp_config = SimulationConfig(entities=entities, issues=issues)
        space_analysis = analyze_negotiation_space(entities, issues, samples=500, rng=temp_config.create_rng())

        for entity in entities:
            # Calculate entity's relative power
            ideal_utility = entity.utility_function.calculate_utility(entity.utility_function.ideal_values)

            # Base parameters
            params = PolicyParameters()

            if space_analysis['has_zopa']:
                zopa_size = space_analysis['zopa_size']

                # Larger ZOPA = can be more demanding
                if zopa_size > 50:
                    params.accept_threshold = 0.75
                    params.initial_demand = 0.95
                else:
                    params.accept_threshold = 0.65
                    params.initial_demand = 0.85

                # Adjust concession rate based on max rounds
                params.concession_rate = 1.0 / (entity.max_rounds * 0.7)

                # Set patience based on ZOPA size
                params.patience = min(40, max(10, zopa_size // 2))

            else:
                # No ZOPA - need to be very flexible
                params.accept_threshold = 0.5
                params.initial_demand = 0.7
                params.concession_rate = 0.15
                params.patience = 10

            suggestions[entity.name] = params

        return suggestions


# ===== LLM INTEGRATION (V2) =====

class LLMAdvisor(NegotiationAdvisor):
    """
    Extended advisor that integrates with LLM APIs for sophisticated analysis.
    This is a template for V2 implementation.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(use_llm=True)
        self.api_key = api_key

    def analyze_with_llm(self,
                        config: SimulationConfig,
                        outcome: NegotiationOutcome,
                        transcript_limit: int = 10) -> str:
        """
        Use LLM to analyze negotiation transcript and provide insights.
        """
        # Prepare transcript for LLM
        transcript_text = self._format_transcript_for_llm(outcome, transcript_limit)

        # Prepare config summary
        config_text = self._format_config_for_llm(config)

        # Create prompt
        prompt = f"""
        Analyze this negotiation outcome and suggest improvements:

        Configuration:
        {config_text}

        Outcome: {'SUCCESS' if outcome.success else f'FAILED - {outcome.impasse_reason}'}
        Rounds: {outcome.rounds_taken}
        Final Utilities: {outcome.final_utilities}

        Recent Transcript:
        {transcript_text}

        Please provide:
        1. Why did this negotiation succeed/fail?
        2. What specific parameter changes would improve outcomes?
        3. What strategic shifts should each party consider?
        4. Key insights about the negotiation dynamics

        Format as JSON with keys: analysis, parameter_changes, strategy_changes, insights
        """

        # In V2, this would call the actual LLM API
        # For now, return a placeholder
        return json.dumps({
            "analysis": "LLM analysis would appear here",
            "parameter_changes": {},
            "strategy_changes": {},
            "insights": []
        })

    def _format_transcript_for_llm(self,
                                   outcome: NegotiationOutcome,
                                   limit: int) -> str:
        """Format negotiation transcript for LLM analysis."""
        lines = []

        for round_obj in outcome.transcript[-limit:]:
            lines.append(f"\nRound {round_obj.round_num}:")
            for offer in round_obj.offers:
                lines.append(f"  {offer.proposer} offers: {offer.values}")
                lines.append(f"  Utilities: {offer.utility_scores}")
                lines.append(f"  Status: {offer.status.value}")
                lines.append(f"  Responses: {round_obj.responses}")

        return "\n".join(lines)

    def _format_config_for_llm(self, config: SimulationConfig) -> str:
        """Format configuration for LLM understanding."""
        lines = []

        lines.append("Entities:")
        for entity in config.entities:
            lines.append(f"  {entity.name}:")
            lines.append(f"    Policy: {entity.policy.type.value}")
            lines.append(f"    Accept Threshold: {entity.policy.params.accept_threshold}")
            lines.append(f"    Weights: {entity.utility_function.weights}")

        lines.append("\nIssues:")
        for issue in config.issues:
            lines.append(f"  {issue.name}: {issue.min_value}-{issue.max_value} {issue.unit or ''}")

        lines.append(f"\nProtocol: {config.protocol}")
        lines.append(f"Max Rounds: {config.max_rounds}")

        return "\n".join(lines)

    def generate_natural_language_strategy(self,
                                          entity: Entity,
                                          opponents: List[Entity],
                                          issues: List[Issue]) -> str:
        """
        Generate natural language negotiation strategy.
        In V2, this would use LLM to create sophisticated strategies.
        """
        # This would call LLM API in V2
        # For now, return template strategy

        strategy_template = f"""
        Negotiation Strategy for {entity.name}:

        Opening Position:
        - Start at {entity.policy.params.initial_demand:.0%} of ideal values
        - Signal flexibility on less important issues

        Concession Strategy:
        - Make small concessions ({entity.policy.params.concession_rate:.0%} per round)
        - Focus concessions on lower-weight issues first

        Red Lines:
        - Do not accept below {entity.policy.params.accept_threshold:.0%} utility
        - Maintain reservation values as absolute minimums

        Tactical Approach:
        - Policy type: {entity.policy.type.value}
        - Patience level: {entity.policy.params.patience} rounds before major moves
        """

        return strategy_template


# ===== HELPER FUNCTIONS =====

def get_advisor(use_llm: bool = False, api_key: Optional[str] = None) -> NegotiationAdvisor:
    """Factory function to get appropriate advisor."""
    if use_llm and api_key:
        return LLMAdvisor(api_key)
    return NegotiationAdvisor(use_llm=False)


def analyze_and_improve(config: SimulationConfig,
                       n_iterations: int = 5) -> Tuple[SimulationConfig, List[NegotiationOutcome]]:
    """
    Iteratively improve negotiation parameters using advisor.
    Returns optimized config and history of outcomes.
    """
    advisor = get_advisor()
    history = []
    current_config = config

    for i in range(n_iterations):
        # Run negotiation
        from protocol import NegotiationEngine
        engine = NegotiationEngine(current_config)
        outcome = engine.run()
        history.append(outcome)

        # Get advice
        report = advisor.analyze_outcome(current_config, outcome, history)

        # Apply parameter suggestions (simplified)
        for suggestion in report.parameter_suggestions:
            if suggestion.confidence > 0.6:  # Only apply high-confidence suggestions
                # In practice, would need proper path-based parameter updating
                for entity in current_config.entities:
                    if entity.name == suggestion.entity_name:
                        if suggestion.parameter_path == "accept_threshold":
                            entity.policy.params.accept_threshold = suggestion.suggested_value
                        elif suggestion.parameter_path == "concession_rate":
                            entity.policy.params.concession_rate = suggestion.suggested_value

        # Check if we should stop
        if outcome.success and all(u > 0.7 for u in outcome.final_utilities.values()):
            break  # Good enough outcome

    return current_config, history
