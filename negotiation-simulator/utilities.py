"""
Utility functions and calculations for the negotiation simulator.
Includes Nash bargaining, Pareto optimality, and other game theory utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from itertools import product, combinations
from models import Entity, Issue, Offer, UtilityFunction


# ===== UTILITY CALCULATIONS =====

def calculate_joint_utility(offer: Dict[str, float],
                           entities: List[Entity]) -> float:
    """Calculate the sum of utilities across all entities."""
    total = 0.0
    for entity in entities:
        total += entity.utility_function.calculate_utility(offer)
    return total


def calculate_nash_product(offer: Dict[str, float],
                          entities: List[Entity]) -> float:
    """
    Calculate Nash bargaining solution product.
    Product of (utility - reservation) for all entities.
    """
    product = 1.0
    for entity in entities:
        utility = entity.utility_function.calculate_utility(offer)
        reservation_utility = 0.0  # Assumes reservation = 0 utility

        # Calculate reservation utility properly
        reservation_offer = {
            issue: entity.utility_function.reservation_values.get(issue, 0)
            for issue in offer.keys()
        }
        reservation_utility = entity.utility_function.calculate_utility(reservation_offer)

        gain = max(0, utility - reservation_utility)
        product *= gain

    return product


def find_nash_bargaining_solution(entities: List[Entity],
                                 issues: List[Issue],
                                 samples: int = 1000) -> Dict[str, float]:
    """
    Find approximate Nash bargaining solution through sampling.
    Returns the offer that maximizes the Nash product.
    """
    best_offer = None
    best_nash_product = -1

    for _ in range(samples):
        # Generate random offer
        offer = {}
        for issue in issues:
            if issue.divisible:
                value = np.random.uniform(issue.min_value, issue.max_value)
            else:
                # For indivisible issues, pick discrete values
                value = np.random.choice([issue.min_value, issue.max_value])
            offer[issue.name] = value

        # Calculate Nash product
        nash_product = calculate_nash_product(offer, entities)

        if nash_product > best_nash_product:
            best_nash_product = nash_product
            best_offer = offer

    return best_offer


# ===== PARETO OPTIMALITY =====

def is_pareto_dominated(offer1: Dict[str, float],
                        offer2: Dict[str, float],
                        entities: List[Entity]) -> bool:
    """
    Check if offer1 is Pareto dominated by offer2.
    offer2 dominates offer1 if it's at least as good for everyone
    and strictly better for at least one entity.
    """
    at_least_as_good = True
    strictly_better_for_someone = False

    for entity in entities:
        u1 = entity.utility_function.calculate_utility(offer1)
        u2 = entity.utility_function.calculate_utility(offer2)

        if u2 < u1:
            at_least_as_good = False
            break
        if u2 > u1:
            strictly_better_for_someone = True

    return at_least_as_good and strictly_better_for_someone


def find_pareto_frontier(offers: List[Dict[str, float]],
                        entities: List[Entity]) -> List[Dict[str, float]]:
    """
    Find the Pareto frontier from a set of offers.
    Returns offers that are not dominated by any other offer.
    """
    frontier = []

    for i, offer in enumerate(offers):
        is_dominated = False
        for j, other_offer in enumerate(offers):
            if i != j and is_pareto_dominated(offer, other_offer, entities):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append(offer)

    return frontier


def is_pareto_optimal(offer: Dict[str, float],
                     entities: List[Entity],
                     issues: List[Issue],
                     samples: int = 100) -> bool:
    """
    Check if an offer is approximately Pareto optimal through sampling.
    """
    current_utilities = [e.utility_function.calculate_utility(offer) for e in entities]

    # Try to find a Pareto improvement
    for _ in range(samples):
        # Generate a nearby offer
        new_offer = {}
        for issue in issues:
            if issue.divisible:
                # Small perturbation
                delta = np.random.normal(0, (issue.max_value - issue.min_value) * 0.1)
                value = offer.get(issue.name, (issue.max_value + issue.min_value) / 2) + delta
                value = np.clip(value, issue.min_value, issue.max_value)
            else:
                value = offer.get(issue.name, issue.min_value)
            new_offer[issue.name] = value

        # Check if this is a Pareto improvement
        new_utilities = [e.utility_function.calculate_utility(new_offer) for e in entities]

        if all(nu >= cu for nu, cu in zip(new_utilities, current_utilities)) and \
           any(nu > cu for nu, cu in zip(new_utilities, current_utilities)):
            return False  # Found a Pareto improvement

    return True  # No Pareto improvement found


# ===== ZONE OF POSSIBLE AGREEMENT (ZOPA) =====

def find_zopa(entities: List[Entity],
              issues: List[Issue],
              samples: int = 1000) -> List[Dict[str, float]]:
    """
    Find the Zone of Possible Agreement (ZOPA).
    Returns offers where all parties get at least their reservation utility.
    """
    zopa = []

    for _ in range(samples):
        # Generate random offer
        offer = {}
        for issue in issues:
            if issue.divisible:
                value = np.random.uniform(issue.min_value, issue.max_value)
            else:
                value = np.random.choice([issue.min_value, issue.max_value])
            offer[issue.name] = value

        # Check if all entities meet their reservation
        all_satisfied = True
        for entity in entities:
            utility = entity.utility_function.calculate_utility(offer)
            if utility < entity.min_acceptable_utility:
                all_satisfied = False
                break

        if all_satisfied:
            zopa.append(offer)

    return zopa


# ===== FAIRNESS METRICS =====

def calculate_kalai_smorodinsky(entities: List[Entity],
                               offer: Dict[str, float]) -> float:
    """
    Calculate distance from Kalai-Smorodinsky solution.
    K-S solution maintains proportional gains from disagreement.
    """
    utilities = []
    max_utilities = []

    for entity in entities:
        utility = entity.utility_function.calculate_utility(offer)
        utilities.append(utility)

        # Find maximum possible utility for this entity
        ideal_offer = {
            issue: entity.utility_function.ideal_values.get(issue, 0)
            for issue in offer.keys()
        }
        max_utility = entity.utility_function.calculate_utility(ideal_offer)
        max_utilities.append(max_utility)

    # Calculate proportionality deviation
    if max(max_utilities) == 0:
        return 0

    ratios = [u / m if m > 0 else 0 for u, m in zip(utilities, max_utilities)]
    deviation = np.std(ratios)

    return deviation


def calculate_egalitarian_score(entities: List[Entity],
                               offer: Dict[str, float]) -> float:
    """
    Calculate egalitarian (maximin) score.
    Maximizes the minimum utility among all parties.
    """
    utilities = [e.utility_function.calculate_utility(offer) for e in entities]
    return min(utilities)


# ===== OFFER GENERATION HELPERS =====

def generate_midpoint_offer(entities: List[Entity],
                           issues: List[Issue]) -> Dict[str, float]:
    """Generate an offer at the midpoint of all ideal values."""
    offer = {}

    for issue in issues:
        ideal_sum = sum(e.utility_function.ideal_values.get(issue.name,
                       (issue.max_value + issue.min_value) / 2)
                       for e in entities)
        offer[issue.name] = ideal_sum / len(entities)

    return offer


def generate_weighted_offer(entities: List[Entity],
                           issues: List[Issue],
                           weights: Dict[str, float]) -> Dict[str, float]:
    """
    Generate offer weighted by entity importance.
    weights: entity_name -> weight
    """
    offer = {}
    total_weight = sum(weights.values())

    for issue in issues:
        weighted_sum = 0
        for entity in entities:
            weight = weights.get(entity.name, 1.0) / total_weight
            ideal = entity.utility_function.ideal_values.get(issue.name,
                   (issue.max_value + issue.min_value) / 2)
            weighted_sum += weight * ideal
        offer[issue.name] = weighted_sum

    return offer


# ===== ANALYSIS FUNCTIONS =====

def analyze_negotiation_space(entities: List[Entity],
                             issues: List[Issue],
                             samples: int = 1000) -> Dict:
    """
    Comprehensive analysis of the negotiation space.
    Returns statistics about ZOPA, Pareto frontier, Nash solution, etc.
    """
    # Find ZOPA
    zopa = find_zopa(entities, issues, samples)

    if not zopa:
        return {
            'has_zopa': False,
            'zopa_size': 0,
            'nash_solution': None,
            'pareto_frontier_size': 0
        }

    # Find Pareto frontier within ZOPA
    pareto_frontier = find_pareto_frontier(zopa[:100], entities)  # Limit for performance

    # Find Nash solution
    nash_solution = find_nash_bargaining_solution(entities, issues, samples)

    # Calculate average utilities in ZOPA
    avg_utilities = {}
    for entity in entities:
        utilities = [entity.utility_function.calculate_utility(offer) for offer in zopa]
        avg_utilities[entity.name] = np.mean(utilities)

    # Normalize ZOPA size to a 0-100 scale for comparability across different sample sizes
    zopa_count = len(zopa)
    zopa_size_scaled = int((zopa_count / max(1, samples)) * 100)

    return {
        'has_zopa': True,
        'zopa_size': zopa_size_scaled,
        'pareto_frontier_size': len(pareto_frontier),
        'nash_solution': nash_solution,
        'nash_product': calculate_nash_product(nash_solution, entities) if nash_solution else 0,
        'average_utilities': avg_utilities,
        'max_joint_utility': max(calculate_joint_utility(o, entities) for o in zopa),
        'min_joint_utility': min(calculate_joint_utility(o, entities) for o in zopa)
    }


def calculate_bargaining_power(entity: Entity,
                              all_entities: List[Entity],
                              issues: List[Issue]) -> float:
    """
    Estimate bargaining power based on:
    - BATNA (reservation values)
    - Patience
    - Issue flexibility
    """
    power_score = 0.0

    # 1. BATNA strength (higher reservation = more power)
    reservation_utility = 0.0
    for issue in issues:
        reservation_offer = {
            issue.name: entity.utility_function.reservation_values.get(issue.name, issue.min_value)
        }
        reservation_utility += entity.utility_function.calculate_utility(reservation_offer)
    power_score += reservation_utility

    # 2. Patience (more patient = more power)
    power_score += entity.policy.params.patience / 100

    # 3. Flexibility (more weight spread = more flexible)
    weights = list(entity.utility_function.weights.values())
    weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights if w > 0)
    power_score += weight_entropy

    # 4. Stubbornness
    power_score += entity.policy.params.stubbornness

    return power_score


# ===== COALITION ANALYSIS =====

def find_stable_coalitions(entities: List[Entity],
                          issues: List[Issue]) -> List[Set[str]]:
    """
    Find coalitions that could form stable agreements.
    Uses simplified core concept from cooperative game theory.
    """
    stable_coalitions = []
    n = len(entities)

    # Check all possible coalitions (power set minus empty set)
    for r in range(2, n + 1):
        for coalition_tuple in combinations(range(n), r):
            coalition = {entities[i].name for i in coalition_tuple}
            coalition_entities = [entities[i] for i in coalition_tuple]

            # Check if coalition has positive surplus
            coalition_offer = generate_midpoint_offer(coalition_entities, issues)

            # Check if all members benefit
            all_benefit = True
            for entity in coalition_entities:
                utility = entity.utility_function.calculate_utility(coalition_offer)
                if utility < entity.min_acceptable_utility:
                    all_benefit = False
                    break

            if all_benefit:
                stable_coalitions.append(coalition)

    return stable_coalitions


# ===== CONCESSION ANALYSIS =====

def calculate_concession_rate(history: List[Offer],
                             entity_name: str) -> float:
    """
    Calculate how fast an entity is conceding based on offer history.
    Returns average utility loss per round.
    """
    entity_offers = [o for o in history if o.proposer == entity_name]

    if len(entity_offers) < 2:
        return 0.0

    utilities = [o.utility_scores.get(entity_name, 0) for o in entity_offers]

    # Calculate average change in utility
    changes = [utilities[i] - utilities[i-1] for i in range(1, len(utilities))]

    return -np.mean(changes) if changes else 0.0  # Negative because concession reduces utility


def predict_convergence_round(entities: List[Entity],
                             history: List[Offer]) -> Optional[int]:
    """
    Predict when negotiation might converge based on concession rates.
    Returns estimated round number or None if no convergence predicted.
    """
    if len(history) < 3:
        return None

    # Calculate concession rates
    rates = {}
    for entity in entities:
        rates[entity.name] = calculate_concession_rate(history, entity.name)

    # Estimate utilities in future rounds
    current_round = len(history)
    max_rounds = 100

    for future_round in range(current_round, max_rounds):
        rounds_ahead = future_round - current_round

        # Project utilities
        projected_utilities = {}
        for entity in entities:
            last_offer = next((o for o in reversed(history) if o.proposer == entity.name), None)
            if last_offer:
                current_utility = last_offer.utility_scores.get(entity.name, 0)
                projected = current_utility - (rates[entity.name] * rounds_ahead)
                projected_utilities[entity.name] = max(0, projected)

        # Check if utilities are converging
        utility_values = list(projected_utilities.values())
        if utility_values and np.std(utility_values) < 0.1:  # Threshold for convergence
            return future_round

    return None
