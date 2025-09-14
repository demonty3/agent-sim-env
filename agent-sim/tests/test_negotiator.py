"""
Unit tests for the negotiation simulator.
Run with: pytest test_negotiation.py -v
"""

import pytest
import numpy as np
from typing import List

from models import (
    Entity, Issue, UtilityFunction, NegotiationPolicy,
    PolicyType, PolicyParameters, SimulationConfig,
    Offer, OfferStatus
)
from protocol import NegotiationEngine, BatchNegotiationRunner
from utilities import (
    calculate_joint_utility, calculate_nash_product,
    is_pareto_dominated, find_zopa, analyze_negotiation_space,
    calculate_concession_rate
)
from advisor import NegotiationAdvisor, get_advisor


# ===== FIXTURES =====

@pytest.fixture
def simple_issues():
    """Create simple test issues."""
    return [
        Issue(name="price", min_value=100, max_value=200, unit="USD"),
        Issue(name="quantity", min_value=10, max_value=100, unit="units")
    ]


@pytest.fixture
def simple_entities(simple_issues):
    """Create simple test entities."""
    buyer = Entity(
        name="Buyer",
        utility_function=UtilityFunction(
            weights={"price": 0.7, "quantity": 0.3},
            ideal_values={"price": 100, "quantity": 100},
            reservation_values={"price": 180, "quantity": 20}
        ),
        policy=NegotiationPolicy(
            type=PolicyType.LINEAR_CONCESSION,
            params=PolicyParameters(accept_threshold=0.6, concession_rate=0.1)
        )
    )

    seller = Entity(
        name="Seller",
        utility_function=UtilityFunction(
            weights={"price": 0.8, "quantity": 0.2},
            ideal_values={"price": 200, "quantity": 10},
            reservation_values={"price": 120, "quantity": 80}
        ),
        policy=NegotiationPolicy(
            type=PolicyType.TIT_FOR_TAT,
            params=PolicyParameters(accept_threshold=0.6)
        )
    )

    return [buyer, seller]


@pytest.fixture
def simple_config(simple_entities, simple_issues):
    """Create simple test configuration."""
    return SimulationConfig(
        entities=simple_entities,
        issues=simple_issues,
        max_rounds=50,
        protocol="alternating"
    )


# ===== MODEL TESTS =====

class TestModels:

    def test_issue_creation(self):
        """Test issue model creation and validation."""
        issue = Issue(name="test", min_value=0, max_value=100)
        assert issue.name == "test"
        assert issue.min_value == 0
        assert issue.max_value == 100
        assert issue.divisible == True  # Default

        # Test validation
        with pytest.raises(ValueError):
            Issue(name="bad", min_value=100, max_value=50)

    def test_utility_calculation(self):
        """Test utility function calculations."""
        utility_fn = UtilityFunction(
            weights={"price": 0.6, "quantity": 0.4},
            ideal_values={"price": 100, "quantity": 50},
            reservation_values={"price": 200, "quantity": 10}
        )

        # Test ideal offer (should give utility close to 1)
        ideal_offer = {"price": 100, "quantity": 50}
        utility = utility_fn.calculate_utility(ideal_offer)
        assert utility == pytest.approx(1.0, rel=0.01)

        # Test reservation offer (should give utility close to 0)
        reservation_offer = {"price": 200, "quantity": 10}
        utility = utility_fn.calculate_utility(reservation_offer)
        assert utility == pytest.approx(0.0, rel=0.01)

        # Test midpoint offer
        mid_offer = {"price": 150, "quantity": 30}
        utility = utility_fn.calculate_utility(mid_offer)
        assert 0 < utility < 1

    def test_policy_offer_generation(self):
        """Test policy offer generation."""
        policy = NegotiationPolicy(
            type=PolicyType.LINEAR_CONCESSION,
            params=PolicyParameters(concession_rate=0.1, initial_demand=0.9)
        )

        utility_fn = UtilityFunction(
            weights={"price": 1.0},
            ideal_values={"price": 100},
            reservation_values={"price": 200}
        )

        issues = [Issue(name="price", min_value=50, max_value=250)]

        # Generate offers at different rounds
        offer_round_1 = policy.make_offer(1, [], utility_fn, issues)
        offer_round_5 = policy.make_offer(5, [], utility_fn, issues)

        assert "price" in offer_round_1
        assert "price" in offer_round_5

        # Later round should have more concession (higher price for buyer)
        assert offer_round_5["price"] > offer_round_1["price"]

    def test_entity_evaluation(self):
        """Test entity offer evaluation."""
        entity = Entity(
            name="TestEntity",
            utility_function=UtilityFunction(
                weights={"price": 1.0},
                ideal_values={"price": 100},
                reservation_values={"price": 200}
            ),
            policy=NegotiationPolicy(
                type=PolicyType.FIXED_THRESHOLD,
                params=PolicyParameters(accept_threshold=0.7)
            )
        )

        # Good offer - should accept
        good_offer = {"price": 130}  # 70% of the way to ideal
        accept, utility = entity.evaluate_offer(good_offer)
        assert accept == True
        assert utility == pytest.approx(0.7, rel=0.01)

        # Bad offer - should reject
        bad_offer = {"price": 170}  # Only 30% utility
        accept, utility = entity.evaluate_offer(bad_offer)
        assert accept == False
        assert utility == pytest.approx(0.3, rel=0.01)


# ===== UTILITY FUNCTION TESTS =====

class TestUtilities:

    def test_joint_utility(self, simple_entities):
        """Test joint utility calculation."""
        offer = {"price": 150, "quantity": 50}
        joint = calculate_joint_utility(offer, simple_entities)

        # Should be sum of individual utilities
        individual_sum = sum(e.utility_function.calculate_utility(offer)
                           for e in simple_entities)
        assert joint == pytest.approx(individual_sum, rel=0.01)

    def test_nash_product(self, simple_entities):
        """Test Nash bargaining product calculation."""
        offer = {"price": 150, "quantity": 50}
        nash_product = calculate_nash_product(offer, simple_entities)

        # Nash product should be non-negative
        assert nash_product >= 0

        # Perfect middle offer should have positive Nash product
        assert nash_product > 0

    def test_pareto_domination(self, simple_entities):
        """Test Pareto domination checking."""
        offer1 = {"price": 150, "quantity": 50}
        offer2 = {"price": 140, "quantity": 60}  # Better for buyer
        offer3 = {"price": 160, "quantity": 40}  # Better for seller

        # offer2 might dominate offer1 for buyer-favorable scenario
        # But shouldn't strictly dominate if seller loses
        is_dominated = is_pareto_dominated(offer1, offer2, simple_entities)

        # Test that same offer doesn't dominate itself
        assert not is_pareto_dominated(offer1, offer1, simple_entities)

    def test_zopa_finding(self, simple_entities, simple_issues):
        """Test ZOPA (Zone of Possible Agreement) finding."""
        zopa = find_zopa(simple_entities, simple_issues, samples=100)

        # Should find some agreements
        assert len(zopa) > 0

        # All ZOPA offers should satisfy minimum requirements
        for offer in zopa:
            for entity in simple_entities:
                utility = entity.utility_function.calculate_utility(offer)
                assert utility >= entity.min_acceptable_utility

    def test_negotiation_space_analysis(self, simple_entities, simple_issues):
        """Test comprehensive space analysis."""
        analysis = analyze_negotiation_space(simple_entities, simple_issues, samples=200)

        assert "has_zopa" in analysis
        assert "zopa_size" in analysis
        assert "nash_solution" in analysis

        # Should have ZOPA for well-configured entities
        assert analysis["has_zopa"] == True
        assert analysis["zopa_size"] > 0

        # Nash solution should exist
        assert analysis["nash_solution"] is not None
        assert isinstance(analysis["nash_solution"], dict)

    def test_concession_rate_calculation(self):
        """Test concession rate calculation."""
        history = [
            Offer(round_num=1, proposer="A", values={"price": 100},
                 utility_scores={"A": 0.9}),
            Offer(round_num=2, proposer="A", values={"price": 110},
                 utility_scores={"A": 0.8}),
            Offer(round_num=3, proposer="A", values={"price": 120},
                 utility_scores={"A": 0.7})
        ]

        rate = calculate_concession_rate(history, "A")

        # Should be positive (conceding = losing utility)
        assert rate > 0
        assert rate == pytest.approx(0.1, rel=0.01)  # Average of 0.1 per round


# ===== PROTOCOL TESTS =====

class TestProtocol:

    def test_negotiation_engine_creation(self, simple_config):
        """Test negotiation engine initialization."""
        engine = NegotiationEngine(simple_config)

        assert engine.current_round == 0
        assert len(engine.entities) == 2
        assert len(engine.issues) == 2
        assert engine.max_rounds == 50

    def test_alternating_protocol(self, simple_config):
        """Test alternating offers protocol."""
        engine = NegotiationEngine(simple_config)
        outcome = engine.run()

        assert outcome is not None
        assert isinstance(outcome.success, bool)
        assert outcome.rounds_taken > 0
        assert outcome.rounds_taken <= simple_config.max_rounds

        if outcome.success:
            assert outcome.final_agreement is not None
            assert all(issue.name in outcome.final_agreement
                      for issue in simple_config.issues)

    def test_simultaneous_protocol(self, simple_entities, simple_issues):
        """Test simultaneous offers protocol."""
        config = SimulationConfig(
            entities=simple_entities,
            issues=simple_issues,
            max_rounds=50,
            protocol="simultaneous"
        )

        engine = NegotiationEngine(config)
        outcome = engine.run()

        assert outcome is not None
        assert outcome.rounds_taken > 0

    def test_deadlock_detection(self):
        """Test that deadlocks are properly detected."""
        # Create entities that won't agree
        entities = [
            Entity(
                name="Stubborn1",
                utility_function=UtilityFunction(
                    weights={"price": 1.0},
                    ideal_values={"price": 100},
                    reservation_values={"price": 150}
                ),
                policy=NegotiationPolicy(
                    type=PolicyType.FIXED_THRESHOLD,
                    params=PolicyParameters(accept_threshold=0.95)  # Very high
                )
            ),
            Entity(
                name="Stubborn2",
                utility_function=UtilityFunction(
                    weights={"price": 1.0},
                    ideal_values={"price": 200},
                    reservation_values={"price": 150}
                ),
                policy=NegotiationPolicy(
                    type=PolicyType.FIXED_THRESHOLD,
                    params=PolicyParameters(accept_threshold=0.95)  # Very high
                )
            )
        ]

        issues = [Issue(name="price", min_value=50, max_value=250)]

        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=100
        )

        engine = NegotiationEngine(config)
        outcome = engine.run()

        assert outcome.success == False
        assert "deadlock" in outcome.impasse_reason.lower() or \
               "max rounds" in outcome.impasse_reason.lower()

    def test_batch_runner(self, simple_config):
        """Test batch simulation runner."""
        runner = BatchNegotiationRunner(simple_config)
        results = runner.run_batch(10)

        assert len(results) == 10
        assert all(isinstance(r.success, bool) for r in results)

        analysis = runner.analyze_results()
        assert "success_rate" in analysis
        assert 0 <= analysis["success_rate"] <= 1
        assert analysis["total_runs"] == 10


# ===== ADVISOR TESTS =====

class TestAdvisor:

    def test_advisor_creation(self):
        """Test advisor initialization."""
        advisor = get_advisor(use_llm=False)
        assert isinstance(advisor, NegotiationAdvisor)
        assert advisor.use_llm == False

    def test_outcome_analysis(self, simple_config):
        """Test advisor outcome analysis."""
        # Run a negotiation
        engine = NegotiationEngine(simple_config)
        outcome = engine.run()

        # Analyze it
        advisor = NegotiationAdvisor()
        report = advisor.analyze_outcome(simple_config, outcome)

        assert report.outcome_analysis is not None
        assert isinstance(report.parameter_suggestions, list)
        assert isinstance(report.strategy_suggestions, list)
        assert isinstance(report.key_insights, list)
        assert 0 <= report.success_probability <= 1

    def test_parameter_suggestions(self, simple_config):
        """Test parameter suggestion generation."""
        # Create a failed negotiation scenario
        for entity in simple_config.entities:
            entity.policy.params.accept_threshold = 0.99  # Too high

        engine = NegotiationEngine(simple_config)
        outcome = engine.run()

        advisor = NegotiationAdvisor()
        report = advisor.analyze_outcome(simple_config, outcome)

        # Should suggest lowering thresholds
        if not outcome.success:
            assert len(report.parameter_suggestions) > 0

            # Check that it suggests lowering threshold
            threshold_suggestions = [s for s in report.parameter_suggestions
                                    if s.parameter_path == "accept_threshold"]
            assert len(threshold_suggestions) > 0

            for suggestion in threshold_suggestions:
                assert suggestion.suggested_value < suggestion.current_value

    def test_success_probability_estimation(self, simple_config):
        """Test success probability estimation."""
        advisor = NegotiationAdvisor()

        # Good config should have high probability
        prob_good = advisor._estimate_success_probability(simple_config, None)
        assert prob_good > 0.5

        # Bad config (no ZOPA) should have zero probability
        for entity in simple_config.entities:
            entity.utility_function.reservation_values = {"price": 150, "quantity": 50}
            entity.utility_function.ideal_values = {"price": 150, "quantity": 50}

        prob_bad = advisor._estimate_success_probability(simple_config, None)
        assert prob_bad == 0.0


# ===== INTEGRATION TESTS =====

class TestIntegration:

    def test_full_negotiation_cycle(self, simple_config):
        """Test complete negotiation cycle with analysis."""
        # Run negotiation
        engine = NegotiationEngine(simple_config)
        outcome = engine.run()

        # Analyze outcome
        advisor = NegotiationAdvisor()
        report = advisor.analyze_outcome(simple_config, outcome)

        # Apply suggestions (if failed)
        if not outcome.success and report.parameter_suggestions:
            for suggestion in report.parameter_suggestions[:2]:  # Apply top 2
                for entity in simple_config.entities:
                    if entity.name == suggestion.entity_name:
                        if suggestion.parameter_path == "accept_threshold":
                            entity.policy.params.accept_threshold = suggestion.suggested_value

            # Re-run with improved parameters
            engine2 = NegotiationEngine(simple_config)
            outcome2 = engine2.run()

            # Second attempt should have better chance
            # (Not guaranteed to succeed, but parameters are improved)
            assert outcome2 is not None

    def test_different_policy_combinations(self, simple_issues):
        """Test various policy combinations."""
        policy_types = [
            PolicyType.LINEAR_CONCESSION,
            PolicyType.TIT_FOR_TAT,
            PolicyType.BOULWARE,
            PolicyType.CONCEDER
        ]

        results = []

        for policy1 in policy_types:
            for policy2 in policy_types:
                entities = [
                    Entity(
                        name="Entity1",
                        utility_function=UtilityFunction(
                            weights={"price": 0.7, "quantity": 0.3},
                            ideal_values={"price": 100, "quantity": 100},
                            reservation_values={"price": 180, "quantity": 20}
                        ),
                        policy=NegotiationPolicy(type=policy1)
                    ),
                    Entity(
                        name="Entity2",
                        utility_function=UtilityFunction(
                            weights={"price": 0.8, "quantity": 0.2},
                            ideal_values={"price": 200, "quantity": 10},
                            reservation_values={"price": 120, "quantity": 80}
                        ),
                        policy=NegotiationPolicy(type=policy2)
                    )
                ]

                config = SimulationConfig(
                    entities=entities,
                    issues=simple_issues,
                    max_rounds=50
                )

                engine = NegotiationEngine(config)
                outcome = engine.run()

                results.append({
                    'policy1': policy1.value,
                    'policy2': policy2.value,
                    'success': outcome.success,
                    'rounds': outcome.rounds_taken
                })

        # Should have tested all combinations
        assert len(results) == len(policy_types) * len(policy_types)

        # At least some combinations should succeed
        successful = [r for r in results if r['success']]
        assert len(successful) > 0

    def test_three_party_negotiation(self):
        """Test three-party negotiation scenario."""
        issues = [
            Issue(name="resource_A", min_value=0, max_value=100),
            Issue(name="resource_B", min_value=0, max_value=100)
        ]

        entities = []
        for i in range(3):
            entities.append(Entity(
                name=f"Party_{i}",
                utility_function=UtilityFunction(
                    weights={"resource_A": 0.5, "resource_B": 0.5},
                    ideal_values={"resource_A": 50 + i*10, "resource_B": 50 - i*10},
                    reservation_values={"resource_A": 20, "resource_B": 20}
                ),
                policy=NegotiationPolicy(
                    type=PolicyType.LINEAR_CONCESSION,
                    params=PolicyParameters(concession_rate=0.1)
                )
            ))

        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=100,
            protocol="simultaneous"  # Better for multi-party
        )

        engine = NegotiationEngine(config)
        outcome = engine.run()

        assert outcome is not None
        assert len(outcome.final_utilities) == 3

        # If successful, all parties should meet minimum utility
        if outcome.success:
            for entity in entities:
                utility = outcome.final_utilities[entity.name]
                assert utility >= entity.min_acceptable_utility


# ===== PERFORMANCE TESTS =====

class TestPerformance:

    def test_large_batch_performance(self, simple_config):
        """Test that batch simulations complete in reasonable time."""
        import time

        start_time = time.time()
        runner = BatchNegotiationRunner(simple_config)
        results = runner.run_batch(100)
        elapsed = time.time() - start_time

        assert len(results) == 100
        assert elapsed < 30  # Should complete 100 simulations in under 30 seconds

        # Check memory usage doesn't explode
        import sys
        total_size = sum(sys.getsizeof(r) for r in results)
        assert total_size < 10 * 1024 * 1024  # Less than 10MB for results

    def test_complex_scenario_performance(self):
        """Test performance with many issues and entities."""
        # Create complex scenario
        issues = [
            Issue(name=f"issue_{i}", min_value=0, max_value=100)
            for i in range(10)  # 10 issues
        ]

        entities = []
        for i in range(5):  # 5 entities
            weights = {f"issue_{j}": np.random.random() for j in range(10)}
            ideal = {f"issue_{j}": np.random.uniform(20, 80) for j in range(10)}
            reservation = {f"issue_{j}": np.random.uniform(10, 90) for j in range(10)}

            entities.append(Entity(
                name=f"Entity_{i}",
                utility_function=UtilityFunction(
                    weights=weights,
                    ideal_values=ideal,
                    reservation_values=reservation
                ),
                policy=NegotiationPolicy(type=PolicyType.LINEAR_CONCESSION)
            ))

        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=50,
            protocol="simultaneous"
        )

        import time
        start_time = time.time()
        engine = NegotiationEngine(config)
        outcome = engine.run()
        elapsed = time.time() - start_time

        assert outcome is not None
        assert elapsed < 5  # Complex negotiation should complete in under 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
