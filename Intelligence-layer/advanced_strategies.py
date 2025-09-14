"""
Advanced negotiation strategies using machine learning and game theory.
These strategies learn from opponent behavior and adapt dynamically.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import random

from models import (
    Entity, Issue, Offer, PolicyType, PolicyParameters,
    NegotiationPolicy, UtilityFunction
)
from utilities import calculate_concession_rate, predict_convergence_round


# ===== ADVANCED STRATEGY COMPONENTS =====

@dataclass
class OpponentModel:
    """Model of opponent's behavior and preferences."""
    name: str
    estimated_weights: Dict[str, float] = field(default_factory=dict)
    estimated_reservation: Dict[str, float] = field(default_factory=dict)
    concession_pattern: str = "unknown"  # linear, aggressive, conservative
    average_concession_rate: float = 0.0
    predictability: float = 0.5  # 0 = random, 1 = perfectly predictable
    offer_history: List[Dict[str, float]] = field(default_factory=list)

    def update(self, offer: Dict[str, float], utility: float):
        """Update model based on new offer."""
        self.offer_history.append(offer)

        # Update concession rate estimate
        if len(self.offer_history) >= 2:
            # Simple moving average of concession
            recent_offers = self.offer_history[-5:]
            if len(recent_offers) >= 2:
                # Estimate based on offer changes
                self._estimate_concession_pattern(recent_offers)

    def _estimate_concession_pattern(self, recent_offers: List[Dict[str, float]]):
        """Estimate opponent's concession pattern."""
        # Simplified pattern detection
        changes = []
        for i in range(1, len(recent_offers)):
            change = sum(abs(recent_offers[i][k] - recent_offers[i-1][k])
                        for k in recent_offers[i].keys())
            changes.append(change)

        if changes:
            avg_change = np.mean(changes)
            std_change = np.std(changes)

            # Classify pattern
            if std_change < avg_change * 0.2:  # Consistent changes
                self.concession_pattern = "linear"
                self.predictability = 0.8
            elif changes[-1] > changes[0]:  # Accelerating
                self.concession_pattern = "aggressive"
                self.predictability = 0.6
            else:  # Decelerating
                self.concession_pattern = "conservative"
                self.predictability = 0.7

            self.average_concession_rate = avg_change


class AdaptiveStrategy(NegotiationPolicy):
    """
    Adaptive strategy that learns from opponent behavior.
    Uses reinforcement learning concepts to adjust parameters.
    """

    def __init__(self, params: Optional[PolicyParameters] = None):
        super().__init__(type=PolicyType.ADAPTIVE, params=params or PolicyParameters())
        self.opponent_models: Dict[str, OpponentModel] = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.memory = deque(maxlen=50)  # Remember last 50 interactions

    def make_offer(self,
                  round_num: int,
                  history: List[Offer],
                  utility_fn: UtilityFunction,
                  issues: List[Issue]) -> Dict[str, float]:
        """Generate offer based on learned opponent behavior."""

        # Update opponent models based on history
        self._update_opponent_models(history)

        # Choose strategy based on opponent behavior
        if round_num <= 3:
            # Initial exploration phase
            return self._exploratory_offer(round_num, utility_fn, issues)

        # Adaptive response based on opponent patterns
        if self._should_exploit():
            return self._exploitation_offer(round_num, history, utility_fn, issues)
        else:
            return self._exploration_offer(round_num, utility_fn, issues)

    def _update_opponent_models(self, history: List[Offer]):
        """Update models of all opponents."""
        for offer in history:
            if offer.proposer not in self.opponent_models:
                self.opponent_models[offer.proposer] = OpponentModel(offer.proposer)

            # Update with offer and utilities
            self.opponent_models[offer.proposer].update(
                offer.values,
                offer.utility_scores.get(offer.proposer, 0)
            )

    def _should_exploit(self) -> bool:
        """Decide whether to exploit learned patterns or explore."""
        return random.random() > self.exploration_rate

    def _exploratory_offer(self,
                          round_num: int,
                          utility_fn: UtilityFunction,
                          issues: List[Issue]) -> Dict[str, float]:
        """Make exploratory offer to learn about opponent."""
        offer = {}

        for issue in issues:
            # Start near ideal but with some randomness
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Add noise for exploration
            noise_factor = 0.1 * (1 - round_num / 100)  # Decrease noise over time
            range_val = abs(ideal - reservation)
            noise = np.random.normal(0, range_val * noise_factor)

            value = ideal - (ideal - reservation) * (round_num * 0.05)
            value = np.clip(value + noise, issue.min_value, issue.max_value)
            offer[issue.name] = value

        return offer

    def _exploitation_offer(self,
                          round_num: int,
                          history: List[Offer],
                          utility_fn: UtilityFunction,
                          issues: List[Issue]) -> Dict[str, float]:
        """Make offer exploiting learned opponent patterns."""

        # Get dominant opponent model
        if self.opponent_models:
            # Find most active opponent
            opponent_activity = {}
            for offer in history[-10:]:  # Recent history
                opponent_activity[offer.proposer] = opponent_activity.get(offer.proposer, 0) + 1

            if opponent_activity:
                main_opponent = max(opponent_activity, key=opponent_activity.get)
                opponent_model = self.opponent_models[main_opponent]

                # Adjust based on opponent's pattern
                if opponent_model.concession_pattern == "aggressive":
                    # Match aggression
                    return self._aggressive_response(round_num, utility_fn, issues)
                elif opponent_model.concession_pattern == "conservative":
                    # Be patient
                    return self._patient_response(round_num, utility_fn, issues)

        # Default adaptive offer
        return self._balanced_offer(round_num, utility_fn, issues)

    def _aggressive_response(self,
                            round_num: int,
                            utility_fn: UtilityFunction,
                            issues: List[Issue]) -> Dict[str, float]:
        """Respond to aggressive opponent with faster concessions."""
        offer = {}
        concession_rate = min(0.15, self.params.concession_rate * 1.5)

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Faster concession
            value = ideal - (ideal - reservation) * (round_num * concession_rate)
            offer[issue.name] = np.clip(value, issue.min_value, issue.max_value)

        return offer

    def _patient_response(self,
                         round_num: int,
                         utility_fn: UtilityFunction,
                         issues: List[Issue]) -> Dict[str, float]:
        """Respond to conservative opponent with patience."""
        offer = {}
        concession_rate = max(0.02, self.params.concession_rate * 0.5)

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Slower concession
            value = ideal - (ideal - reservation) * (round_num * concession_rate)
            offer[issue.name] = np.clip(value, issue.min_value, issue.max_value)

        return offer

    def _balanced_offer(self,
                       round_num: int,
                       utility_fn: UtilityFunction,
                       issues: List[Issue]) -> Dict[str, float]:
        """Make balanced offer."""
        offer = {}

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Standard concession
            value = ideal - (ideal - reservation) * (round_num * self.params.concession_rate)
            offer[issue.name] = np.clip(value, issue.min_value, issue.max_value)

        return offer


class MixedStrategy(NegotiationPolicy):
    """
    Mixed strategy that probabilistically chooses between different strategies.
    Based on game theory mixed strategy equilibrium concepts.
    """

    def __init__(self,
                strategy_weights: Optional[Dict[PolicyType, float]] = None,
                params: Optional[PolicyParameters] = None):
        super().__init__(type=PolicyType.ADAPTIVE, params=params or PolicyParameters())

        # Default mixed strategy
        self.strategy_weights = strategy_weights or {
            PolicyType.LINEAR_CONCESSION: 0.4,
            PolicyType.TIT_FOR_TAT: 0.3,
            PolicyType.BOULWARE: 0.2,
            PolicyType.CONCEDER: 0.1
        }

        # Normalize weights
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v/total for k, v in self.strategy_weights.items()}

        # Track performance of each strategy
        self.strategy_performance: Dict[PolicyType, float] = {
            k: 0.5 for k in self.strategy_weights.keys()
        }

        self.current_strategy: Optional[PolicyType] = None
        self.strategy_rounds = 0
        self.switch_threshold = 5  # Rounds before considering switch

    def make_offer(self,
                  round_num: int,
                  history: List[Offer],
                  utility_fn: UtilityFunction,
                  issues: List[Issue]) -> Dict[str, float]:
        """Generate offer using mixed strategy."""

        # Decide whether to switch strategies
        if self.strategy_rounds >= self.switch_threshold or self.current_strategy is None:
            self.current_strategy = self._select_strategy()
            self.strategy_rounds = 0

        self.strategy_rounds += 1

        # Execute current strategy
        if self.current_strategy == PolicyType.LINEAR_CONCESSION:
            return self._linear_concession_offer(round_num, utility_fn, issues)
        elif self.current_strategy == PolicyType.TIT_FOR_TAT:
            return self._tit_for_tat_offer(round_num, history, utility_fn, issues)
        elif self.current_strategy == PolicyType.BOULWARE:
            return self._boulware_offer(round_num, utility_fn, issues)
        elif self.current_strategy == PolicyType.CONCEDER:
            return self._conceder_offer(round_num, utility_fn, issues)
        else:
            return self._linear_concession_offer(round_num, utility_fn, issues)

    def _select_strategy(self) -> PolicyType:
        """Select strategy based on weights and performance."""
        # Adjust weights based on performance
        adjusted_weights = {}
        for strategy, base_weight in self.strategy_weights.items():
            performance_factor = self.strategy_performance.get(strategy, 0.5)
            adjusted_weights[strategy] = base_weight * (0.5 + performance_factor)

        # Normalize
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}

        # Probabilistic selection
        r = random.random()
        cumulative = 0
        for strategy, weight in adjusted_weights.items():
            cumulative += weight
            if r <= cumulative:
                return strategy

        return PolicyType.LINEAR_CONCESSION  # Fallback

    def _linear_concession_offer(self, round_num: int,
                                utility_fn: UtilityFunction,
                                issues: List[Issue]) -> Dict[str, float]:
        """Linear concession strategy."""
        offer = {}
        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            progress = min(1.0, round_num * self.params.concession_rate)
            value = ideal - progress * (ideal - reservation)
            offer[issue.name] = value

        return offer

    def _tit_for_tat_offer(self, round_num: int,
                          history: List[Offer],
                          utility_fn: UtilityFunction,
                          issues: List[Issue]) -> Dict[str, float]:
        """Tit-for-tat strategy - mirror opponent's concessions."""
        if len(history) < 2:
            return self._linear_concession_offer(round_num, utility_fn, issues)

        # Calculate opponent's last concession
        opponent_concession = 0
        for i in range(len(history) - 1, 0, -1):
            if history[i].proposer != history[i-1].proposer:
                # Found opponent's last two offers
                for issue in issues:
                    if issue.name in history[i].values and issue.name in history[i-1].values:
                        opponent_concession += abs(
                            history[i].values[issue.name] - history[i-1].values[issue.name]
                        )
                break

        # Mirror the concession
        offer = {}
        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Apply similar concession magnitude
            concession_factor = min(1.0, opponent_concession / (len(issues) * 10))
            value = ideal - (ideal - reservation) * concession_factor * round_num * 0.1
            offer[issue.name] = np.clip(value, issue.min_value, issue.max_value)

        return offer

    def _boulware_offer(self, round_num: int,
                       utility_fn: UtilityFunction,
                       issues: List[Issue]) -> Dict[str, float]:
        """Boulware strategy - hard early, concede late."""
        offer = {}
        max_rounds = 100  # Assumed max

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Exponential concession curve
            progress = (round_num / max_rounds) ** 3  # Cubic for late concession
            value = ideal - progress * (ideal - reservation)
            offer[issue.name] = value

        return offer

    def _conceder_offer(self, round_num: int,
                       utility_fn: UtilityFunction,
                       issues: List[Issue]) -> Dict[str, float]:
        """Conceder strategy - concede early, hard late."""
        offer = {}
        max_rounds = 100

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Inverse exponential - fast early concession
            progress = 1 - (1 - round_num / max_rounds) ** 3
            value = ideal - progress * (ideal - reservation)
            offer[issue.name] = value

        return offer

    def update_performance(self, strategy: PolicyType, outcome_utility: float):
        """Update strategy performance based on outcome."""
        # Exponential moving average
        alpha = 0.2
        old_performance = self.strategy_performance.get(strategy, 0.5)
        self.strategy_performance[strategy] = alpha * outcome_utility + (1 - alpha) * old_performance


class MonteCarloTreeSearchStrategy(NegotiationPolicy):
    """
    MCTS-based strategy that simulates future negotiation paths.
    Particularly effective for multi-issue, multi-party negotiations.
    """

    def __init__(self,
                simulations: int = 100,
                params: Optional[PolicyParameters] = None):
        super().__init__(type=PolicyType.ADAPTIVE, params=params or PolicyParameters())
        self.simulations = simulations
        self.exploration_constant = 1.414  # sqrt(2) for UCB1

    def make_offer(self,
                  round_num: int,
                  history: List[Offer],
                  utility_fn: UtilityFunction,
                  issues: List[Issue]) -> Dict[str, float]:
        """Generate offer using MCTS lookahead."""

        # Generate candidate offers
        candidates = self._generate_candidates(utility_fn, issues, n=10)

        # Simulate outcomes for each candidate
        best_offer = None
        best_value = -float('inf')

        for candidate in candidates:
            # Run simulations
            total_value = 0
            for _ in range(self.simulations // len(candidates)):
                simulated_value = self._simulate_negotiation(
                    candidate, round_num, history, utility_fn, issues
                )
                total_value += simulated_value

            avg_value = total_value / (self.simulations // len(candidates))

            if avg_value > best_value:
                best_value = avg_value
                best_offer = candidate

        return best_offer or candidates[0]

    def _generate_candidates(self,
                            utility_fn: UtilityFunction,
                            issues: List[Issue],
                            n: int = 10) -> List[Dict[str, float]]:
        """Generate candidate offers to evaluate."""
        candidates = []

        for i in range(n):
            offer = {}
            for issue in issues:
                ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
                reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

                # Generate offers at different concession levels
                concession_level = i / n
                value = ideal - concession_level * (ideal - reservation)
                offer[issue.name] = value

            candidates.append(offer)

        return candidates

    def _simulate_negotiation(self,
                            initial_offer: Dict[str, float],
                            round_num: int,
                            history: List[Offer],
                            utility_fn: UtilityFunction,
                            issues: List[Issue]) -> float:
        """Simulate negotiation from given offer."""
        # Simplified simulation - estimates outcome value
        utility = utility_fn.calculate_utility(initial_offer)

        # Estimate opponent acceptance probability
        if len(history) > 0:
            # Based on historical patterns
            recent_utilities = [o.utility_scores.get(o.proposer, 0)
                              for o in history[-5:]]
            if recent_utilities:
                avg_opponent_demand = np.mean(recent_utilities)
                acceptance_prob = 1 / (1 + np.exp(-10 * (utility - avg_opponent_demand)))
            else:
                acceptance_prob = 0.5
        else:
            acceptance_prob = 0.5

        # Expected value = utility * acceptance probability
        return utility * acceptance_prob


class ReinforcementLearningStrategy(NegotiationPolicy):
    """
    Q-learning based strategy that learns optimal actions through experience.
    """

    def __init__(self,
                alpha: float = 0.1,  # Learning rate
                gamma: float = 0.9,  # Discount factor
                epsilon: float = 0.1,  # Exploration rate
                params: Optional[PolicyParameters] = None):
        super().__init__(type=PolicyType.ADAPTIVE, params=params or PolicyParameters())
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: (state, action) -> value
        self.q_table: Dict[Tuple[str, str], float] = {}

        # State discretization parameters
        self.round_buckets = 10
        self.utility_buckets = 10

    def make_offer(self,
                  round_num: int,
                  history: List[Offer],
                  utility_fn: UtilityFunction,
                  issues: List[Issue]) -> Dict[str, float]:
        """Generate offer using Q-learning policy."""

        # Get current state
        state = self._get_state(round_num, history, utility_fn)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            action = self._random_action(utility_fn, issues)
        else:
            # Exploit: best known action
            action = self._best_action(state, utility_fn, issues)

        # Convert action to offer
        offer = self._action_to_offer(action, utility_fn, issues)

        # Update Q-value based on history if available
        if len(history) > 0:
            self._update_q_values(history, utility_fn)

        return offer

    def _get_state(self,
                  round_num: int,
                  history: List[Offer],
                  utility_fn: UtilityFunction) -> str:
        """Convert current situation to discrete state."""
        # Discretize round number
        round_bucket = min(round_num // 10, self.round_buckets - 1)

        # Discretize current utility level
        if history:
            last_offer = history[-1]
            current_utility = utility_fn.calculate_utility(last_offer.values)
            utility_bucket = int(current_utility * self.utility_buckets)
        else:
            utility_bucket = self.utility_buckets - 1  # Start at high utility

        return f"r{round_bucket}_u{utility_bucket}"

    def _random_action(self,
                      utility_fn: UtilityFunction,
                      issues: List[Issue]) -> str:
        """Generate random action."""
        actions = ["aggressive", "moderate", "conservative", "hold"]
        return random.choice(actions)

    def _best_action(self,
                    state: str,
                    utility_fn: UtilityFunction,
                    issues: List[Issue]) -> str:
        """Select best action based on Q-values."""
        actions = ["aggressive", "moderate", "conservative", "hold"]

        best_action = actions[0]
        best_value = -float('inf')

        for action in actions:
            q_value = self.q_table.get((state, action), 0)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def _action_to_offer(self,
                        action: str,
                        utility_fn: UtilityFunction,
                        issues: List[Issue]) -> Dict[str, float]:
        """Convert discrete action to continuous offer."""
        offer = {}

        # Map action to concession level
        concession_levels = {
            "aggressive": 0.15,
            "moderate": 0.08,
            "conservative": 0.03,
            "hold": 0.0
        }

        concession = concession_levels.get(action, 0.05)

        for issue in issues:
            ideal = utility_fn.ideal_values.get(issue.name, issue.max_value)
            reservation = utility_fn.reservation_values.get(issue.name, issue.min_value)

            # Apply concession
            value = ideal - concession * (ideal - reservation)
            offer[issue.name] = np.clip(value, issue.min_value, issue.max_value)

        return offer

    def _update_q_values(self,
                        history: List[Offer],
                        utility_fn: UtilityFunction):
        """Update Q-values based on observed outcomes."""
        if len(history) < 2:
            return

        # Get previous state-action pair
        prev_round = len(history) - 1
        prev_state = self._get_state(prev_round, history[:-1], utility_fn)

        # Estimate action taken (simplified)
        if len(history) >= 2:
            utility_change = (utility_fn.calculate_utility(history[-1].values) -
                            utility_fn.calculate_utility(history[-2].values))

            if utility_change < -0.1:
                action = "aggressive"
            elif utility_change < -0.05:
                action = "moderate"
            elif utility_change < 0:
                action = "conservative"
            else:
                action = "hold"

            # Calculate reward (simplified)
            current_utility = utility_fn.calculate_utility(history[-1].values)
            reward = current_utility

            # Update Q-value
            old_q = self.q_table.get((prev_state, action), 0)

            # Get max Q-value for next state
            current_state = self._get_state(len(history), history, utility_fn)
            max_next_q = max([self.q_table.get((current_state, a), 0)
                            for a in ["aggressive", "moderate", "conservative", "hold"]])

            # Q-learning update
            new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
            self.q_table[(prev_state, action)] = new_q


# ===== UTILITY FUNCTIONS =====

def create_advanced_strategy(strategy_type: str,
                           **kwargs) -> NegotiationPolicy:
    """Factory function for creating advanced strategies."""

    if strategy_type == "adaptive":
        return AdaptiveStrategy(kwargs.get('params'))
    elif strategy_type == "mixed":
        return MixedStrategy(
            kwargs.get('strategy_weights'),
            kwargs.get('params')
        )
    elif strategy_type == "mcts":
        return MonteCarloTreeSearchStrategy(
            kwargs.get('simulations', 100),
            kwargs.get('params')
        )
    elif strategy_type == "q_learning":
        return ReinforcementLearningStrategy(
            kwargs.get('alpha', 0.1),
            kwargs.get('gamma', 0.9),
            kwargs.get('epsilon', 0.1),
            kwargs.get('params')
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def compare_strategies(entities: List[Entity],
                      issues: List[Issue],
                      strategies: List[str],
                      n_runs: int = 100) -> pd.DataFrame:
    """
    Compare performance of different strategies.
    Returns DataFrame with comparison metrics.
    """
    from protocol import NegotiationEngine
    from models import SimulationConfig

    results = []

    for strategy in strategies:
        # Create entities with this strategy
        test_entities = []
        for entity in entities:
            new_entity = Entity(
                name=entity.name,
                utility_function=entity.utility_function,
                policy=create_advanced_strategy(strategy)
            )
            test_entities.append(new_entity)

        # Run simulations
        config = SimulationConfig(
            entities=test_entities,
            issues=issues,
            max_rounds=100
        )

        strategy_results = []
        for _ in range(n_runs):
            engine = NegotiationEngine(config)
            outcome = engine.run()

            strategy_results.append({
                'success': outcome.success,
                'rounds': outcome.rounds_taken,
                'avg_utility': np.mean(list(outcome.final_utilities.values())) if outcome.success else 0
            })

        # Aggregate results
        df = pd.DataFrame(strategy_results)
        results.append({
            'Strategy': strategy,
            'Success Rate': df['success'].mean(),
            'Avg Rounds': df['rounds'].mean(),
            'Avg Utility': df['avg_utility'].mean(),
            'Std Utility': df['avg_utility'].std()
        })

    return pd.DataFrame(results)
