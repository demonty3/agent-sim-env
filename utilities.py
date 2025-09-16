"""
Utility functions and calculations for the negotiation simulator.
Includes Nash bargaining, Pareto optimality, and other game theory utilities.
"""

import numpy as np
from numpy.random import Generator, default_rng
from typing import Dict, List, Tuple, Optional, Set
from itertools import product, combinations
from models import Entity, Issue, Offer, UtilityFunction


# ===== UTILITY CALCULATIONS =====

def calculate_joint_utility(offer: Dict[str, float],
                           entities: List[Entity]) -> float:
    """Compute the total utility that a set of entities assigns to an offer.

    Args:
        offer: Proposed values keyed by issue name.
        entities: Negotiating entities whose utility functions will be
            evaluated.

    Returns:
        float: Sum of individual utilities across all provided entities.

    Side Effects:
        None.
    """

    total = 0.0
    for entity in entities:
        total += entity.utility_function.calculate_utility(offer)
    return total


def calculate_nash_product(offer: Dict[str, float],
                          entities: List[Entity]) -> float:
    """Evaluate the Nash bargaining product for a proposed offer.

    Args:
        offer: Candidate agreement described as issue-to-value mapping.
        entities: Participants whose utilities and reservation values inform
            the Nash product.

    Returns:
        float: Product of the utility gains above reservation utilities across
        all entities.

    Side Effects:
        None.
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


def find_nash_bargaining_solution(
    entities: List[Entity],
    issues: List[Issue],
    samples: int = 1000,
    rng: Optional[Generator] = None,
) -> Dict[str, float]:
    """Approximate the Nash bargaining solution via random sampling.

    Returns the sampled offer that maximizes the Nash product.
    """
    rng = default_rng() if rng is None else rng
    best_offer = None
    best_nash_product = -1

    for _ in range(samples):
        # Generate random offer
        offer = {}
        for issue in issues:
            if issue.divisible:
                value = rng.uniform(issue.min_value, issue.max_value)
            else:
                # For indivisible issues, pick discrete values
                value = rng.choice([issue.min_value, issue.max_value])
            offer[issue.name] = float(value)

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
    """Determine whether ``offer1`` is Pareto dominated by ``offer2``.

    Args:
        offer1: Baseline offer to test for dominance.
        offer2: Candidate offer that may dominate ``offer1``.
        entities: Participants whose utilities are compared.

    Returns:
        bool: ``True`` if ``offer2`` is at least as good for every entity and
        strictly better for at least one entity.

    Side Effects:
        None.
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
    """Extract the subset of non-dominated offers from a candidate set.

    Args:
        offers: Offers to evaluate for Pareto dominance.
        entities: Participants whose utilities define dominance.

    Returns:
        List[Dict[str, float]]: Offers that are not dominated by any other
        offer in the provided list.

    Side Effects:
        None.
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


def is_pareto_optimal(
    offer: Dict[str, float],
    entities: List[Entity],
    issues: List[Issue],
    samples: int = 100,
    rng: Optional[Generator] = None,
) -> bool:
    """Assess Pareto optimality of an offer via randomized search."""
    rng = default_rng() if rng is None else rng
    current_utilities = [e.utility_function.calculate_utility(offer) for e in entities]

    # Try to find a Pareto improvement
    for _ in range(samples):
        # Generate a nearby offer
        new_offer = {}
        for issue in issues:
            if issue.divisible:
                # Small perturbation
                delta = rng.normal(0, (issue.max_value - issue.min_value) * 0.1)
                value = offer.get(issue.name, (issue.max_value + issue.min_value) / 2) + delta
                value = np.clip(value, issue.min_value, issue.max_value)
            else:
                value = offer.get(issue.name, issue.min_value)
            new_offer[issue.name] = float(value)

        # Check if this is a Pareto improvement
        new_utilities = [e.utility_function.calculate_utility(new_offer) for e in entities]

        if (
            all(nu >= cu for nu, cu in zip(new_utilities, current_utilities))
            and any(nu > cu for nu, cu in zip(new_utilities, current_utilities))
        ):
            return False  # Found a Pareto improvement

    return True  # No Pareto improvement found


# ===== ZONE OF POSSIBLE AGREEMENT (ZOPA) =====

def find_zopa(
    entities: List[Entity],
    issues: List[Issue],
    samples: int = 1000,
    rng: Optional[Generator] = None,
) -> List[Dict[str, float]]:
    """Estimate the Zone of Possible Agreement (ZOPA) by random sampling."""
    rng = default_rng() if rng is None else rng
    zopa = []

    for _ in range(samples):
        # Generate random offer
        offer = {}
        for issue in issues:
            if issue.divisible:
                value = rng.uniform(issue.min_value, issue.max_value)
            else:
                value = rng.choice([issue.min_value, issue.max_value])
            offer[issue.name] = float(value)

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
    """Measure deviation from the Kalai-Smorodinsky proportionality ideal.

    Args:
        entities: Negotiating participants whose utilities are analyzed.
        offer: Candidate offer whose fairness is being measured.

    Returns:
        float: Standard deviation of proportional gains; lower values indicate
        offers closer to the Kalai-Smorodinsky solution.

    Side Effects:
        None.
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
    """Compute the egalitarian score (maximin utility) for an offer.

    Args:
        entities: Negotiating entities whose utilities are considered.
        offer: Offer used to evaluate fairness.

    Returns:
        float: Minimum utility value across all entities.

    Side Effects:
        None.
    """

    utilities = [e.utility_function.calculate_utility(offer) for e in entities]
    return min(utilities)


# ===== OFFER GENERATION HELPERS =====

def generate_midpoint_offer(entities: List[Entity],
                           issues: List[Issue]) -> Dict[str, float]:
    """Construct an offer at the midpoint of participants' ideal values.

    Args:
        entities: Entities contributing ideal values for each issue.
        issues: Issues that must appear in the generated offer.

    Returns:
        Dict[str, float]: Offer with each issue set to the average of entity
        ideal values.

    Side Effects:
        None.
    """

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
    """Create an offer that respects externally supplied entity weights.

    Args:
        entities: Entities contributing their ideal values.
        issues: Issues to include in the generated offer.
        weights: Mapping from entity name to relative influence weight.

    Returns:
        Dict[str, float]: Offer where each issue value is a weighted average of
        entity ideals.

    Side Effects:
        None.
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

def analyze_negotiation_space(
    entities: List[Entity],
    issues: List[Issue],
    samples: int = 1000,
    rng: Optional[Generator] = None,
) -> Dict:
    """Collect coarse statistics about the negotiation landscape."""
    rng = default_rng() if rng is None else rng
    # Find ZOPA
    zopa = find_zopa(entities, issues, samples, rng=rng)

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
    nash_solution = find_nash_bargaining_solution(entities, issues, samples, rng=rng)

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
    """Estimate an entity's bargaining power from several heuristics.

    Args:
        entity: Entity whose bargaining power is being scored.
        all_entities: All participants, currently unused but allows contextual
            extensions.
        issues: Issues that determine reservation utilities and flexibility.

    Returns:
        float: Composite power score where higher values imply stronger
        bargaining positions.

    Side Effects:
        None.
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
    """Identify coalitions whose members all benefit from collaborating.

    Args:
        entities: All available entities that may form coalitions.
        issues: Issues considered when evaluating coalition offers.

    Returns:
        List[Set[str]]: Coalitions where every member achieves at least their
        minimum acceptable utility.

    Side Effects:
        None.
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
    """Estimate the average utility loss per round for an entity.

    Args:
        history: Complete offer history from a negotiation.
        entity_name: Name of the entity whose concessions are analyzed.

    Returns:
        float: Average decrease in utility between consecutive offers made by
        the entity. Returns ``0.0`` when insufficient data is available.

    Side Effects:
        None.
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
    """Forecast the round where utilities may converge across entities.

    Args:
        entities: Participants whose concession rates are examined.
        history: Full list of offers exchanged during the negotiation.

    Returns:
        Optional[int]: Estimated future round number where utilities align, or
        ``None`` when convergence cannot be inferred.

    Side Effects:
        None.
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
