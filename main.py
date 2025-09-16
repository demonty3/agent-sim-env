"""
Main entry point for the negotiation simulator.
Run this file to see a quick demonstration or import for use in other scripts.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from models import (
    Entity, Issue, UtilityFunction, NegotiationPolicy,
    PolicyType, PolicyParameters, SimulationConfig
)
from protocol import NegotiationEngine, BatchNegotiationRunner
from utilities import analyze_negotiation_space, find_nash_bargaining_solution

console = Console()


def create_simple_negotiation():
    """Create a simple two-party negotiation scenario."""

    # Define the issues to negotiate
    issues = [
        Issue(
            name="price",
            min_value=1000,
            max_value=5000,
            unit="USD"
        ),
        Issue(
            name="delivery_time",
            min_value=1,
            max_value=30,
            unit="days"
        ),
        Issue(
            name="quantity",
            min_value=100,
            max_value=1000,
            unit="units"
        )
    ]

    # Create buyer entity
    buyer = Entity(
        name="Buyer",
        type="company",
        utility_function=UtilityFunction(
            weights={
                "price": 0.5,      # Price is important
                "delivery_time": 0.3,  # Delivery matters
                "quantity": 0.2    # Quantity less important
            },
            ideal_values={
                "price": 1000,     # Wants low price
                "delivery_time": 1,    # Wants fast delivery
                "quantity": 1000   # Wants high quantity
            },
            reservation_values={
                "price": 4000,     # Won't pay more than this
                "delivery_time": 20,   # Can't wait longer
                "quantity": 200    # Minimum acceptable quantity
            }
        ),
        policy=NegotiationPolicy(
            type=PolicyType.LINEAR_CONCESSION,
            params=PolicyParameters(
                accept_threshold=0.65,
                initial_demand=0.9,
                concession_rate=0.08,
                patience=15
            )
        ),
        min_acceptable_utility=0.4
    )

    # Create seller entity
    seller = Entity(
        name="Seller",
        type="company",
        utility_function=UtilityFunction(
            weights={
                "price": 0.6,      # Price most important
                "delivery_time": 0.2,  # Delivery less important
                "quantity": 0.2    # Quantity matters
            },
            ideal_values={
                "price": 5000,     # Wants high price
                "delivery_time": 30,   # Prefers more time
                "quantity": 100    # Prefers smaller orders
            },
            reservation_values={
                "price": 2000,     # Won't sell for less
                "delivery_time": 5,    # Can't deliver faster
                "quantity": 800    # Maximum capacity
            }
        ),
        policy=NegotiationPolicy(
            type=PolicyType.TIT_FOR_TAT,
            params=PolicyParameters(
                accept_threshold=0.6,
                initial_demand=0.85,
                stubbornness=0.4,
                patience=20
            )
        ),
        min_acceptable_utility=0.45
    )

    # Create configuration
    config = SimulationConfig(
        entities=[buyer, seller],
        issues=issues,
        max_rounds=50,
        protocol="alternating",
        track_pareto=True,
        calculate_nash=True
    )

    return config


def create_complex_negotiation():
    """Create a more complex multi-party resource allocation scenario."""

    # Define resources to allocate
    issues = [
        Issue(name="water_rights", min_value=0, max_value=100, unit="million_gallons"),
        Issue(name="energy_allocation", min_value=0, max_value=100, unit="GWh"),
        Issue(name="land_use", min_value=0, max_value=100, unit="sq_km"),
        Issue(name="carbon_credits", min_value=0, max_value=1000, unit="tons")
    ]

    # Country A: Industrial focus
    country_a = Entity(
        name="Industrial_Nation",
        type="country",
        utility_function=UtilityFunction(
            weights={
                "water_rights": 0.2,
                "energy_allocation": 0.4,
                "land_use": 0.3,
                "carbon_credits": 0.1
            },
            ideal_values={
                "water_rights": 40,
                "energy_allocation": 80,
                "land_use": 60,
                "carbon_credits": 200
            },
            reservation_values={
                "water_rights": 20,
                "energy_allocation": 40,
                "land_use": 30,
                "carbon_credits": 100
            }
        ),
        policy=NegotiationPolicy(
            type=PolicyType.BOULWARE,  # Hard negotiator
            params=PolicyParameters(
                accept_threshold=0.7,
                initial_demand=0.95,
                concession_rate=0.03,
                patience=40,
                stubbornness=0.7
            )
        )
    )

    # Country B: Agricultural focus
    country_b = Entity(
        name="Agricultural_Nation",
        type="country",
        utility_function=UtilityFunction(
            weights={
                "water_rights": 0.5,
                "energy_allocation": 0.1,
                "land_use": 0.3,
                "carbon_credits": 0.1
            },
            ideal_values={
                "water_rights": 70,
                "energy_allocation": 30,
                "land_use": 80,
                "carbon_credits": 300
            },
            reservation_values={
                "water_rights": 35,
                "energy_allocation": 15,
                "land_use": 40,
                "carbon_credits": 150
            }
        ),
        policy=NegotiationPolicy(
            type=PolicyType.CONCEDER,  # More willing to compromise
            params=PolicyParameters(
                accept_threshold=0.6,
                initial_demand=0.8,
                concession_rate=0.12,
                patience=25
            )
        )
    )

    # Country C: Balanced approach
    country_c = Entity(
        name="Balanced_Nation",
        type="country",
        utility_function=UtilityFunction(
            weights={
                "water_rights": 0.25,
                "energy_allocation": 0.25,
                "land_use": 0.25,
                "carbon_credits": 0.25
            },
            ideal_values={
                "water_rights": 50,
                "energy_allocation": 50,
                "land_use": 50,
                "carbon_credits": 500
            },
            reservation_values={
                "water_rights": 25,
                "energy_allocation": 25,
                "land_use": 25,
                "carbon_credits": 250
            }
        ),
        policy=NegotiationPolicy(
            type=PolicyType.ADAPTIVE,
            params=PolicyParameters(
                accept_threshold=0.65,
                initial_demand=0.85,
                concession_rate=0.08,
                learning_rate=0.15,
                exploration_factor=0.2
            )
        )
    )

    config = SimulationConfig(
        entities=[country_a, country_b, country_c],
        issues=issues,
        max_rounds=100,
        protocol="simultaneous",  # All propose at once
        track_pareto=True,
        calculate_nash=True
    )

    return config


def run_demo():
    """Run a demonstration of the negotiation simulator."""

    # Header
    console.print(Panel.fit(
        Text("AI Agent Negotiation Simulator", style="bold cyan"),
        subtitle="Demo Mode"
    ))

    console.print("\n[yellow]Choose a scenario:[/yellow]")
    console.print("1. Simple buyer-seller negotiation")
    console.print("2. Complex multi-party resource allocation")
    console.print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        config = create_simple_negotiation()
        scenario_name = "Buyer-Seller Negotiation"
    elif choice == "2":
        config = create_complex_negotiation()
        scenario_name = "Multi-Party Resource Allocation"
    else:
        console.print("[red]Exiting...[/red]")
        return

    # Display scenario
    console.print(f"\n[bold cyan]Running: {scenario_name}[/bold cyan]")
    console.print(f"Entities: {', '.join([e.name for e in config.entities])}")
    console.print(f"Issues: {', '.join([i.name for i in config.issues])}")
    console.print(f"Protocol: {config.protocol}")
    console.print(f"Max rounds: {config.max_rounds}")

    # Pre-negotiation analysis
    console.print("\n[yellow]Analyzing negotiation space...[/yellow]")
    analysis = analyze_negotiation_space(config.entities, config.issues, samples=500)

    if not analysis['has_zopa']:
        console.print("[red]❌ Warning: No Zone of Possible Agreement exists![/red]")
        console.print("The parties' requirements are incompatible.")
    else:
        console.print(f"[green]✅ ZOPA exists with {analysis['zopa_size']} possible agreements[/green]")

    # Run negotiation
    console.print("\n[yellow]Starting negotiation...[/yellow]")
    engine = NegotiationEngine(config)
    outcome = engine.run()

    # Display results
    console.print(f"\n[bold]{outcome.summary()}[/bold]")

    if outcome.success:
        console.print("\n[green]Final Agreement:[/green]")
        for issue, value in outcome.final_agreement.items():
            # Find the issue object to get unit
            issue_obj = next((i for i in config.issues if i.name == issue), None)
            unit = f" {issue_obj.unit}" if issue_obj and issue_obj.unit else ""
            console.print(f"  {issue}: {value:.2f}{unit}")

        console.print("\n[cyan]Entity Satisfaction (Utility Scores):[/cyan]")
        for entity, utility in outcome.final_utilities.items():
            percentage = utility * 100
            bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
            console.print(f"  {entity:20} {bar} {percentage:.1f}%")

        # Analysis
        if outcome.pareto_optimal is not None:
            status = "✅ Yes" if outcome.pareto_optimal else "❌ No (could be improved)"
            console.print(f"\n[cyan]Pareto Optimal:[/cyan] {status}")

        if outcome.nash_bargaining_score is not None:
            console.print(f"[cyan]Nash Product Score:[/cyan] {outcome.nash_bargaining_score:.3f}")

        # Show negotiation dynamics
        console.print(f"\n[cyan]Negotiation Dynamics:[/cyan]")
        console.print(f"  Total rounds: {outcome.rounds_taken}")

        # Calculate average concession rates
        if outcome.transcript:
            from utilities import calculate_concession_rate
            flat_history = []
            for round_obj in outcome.transcript:
                flat_history.extend(round_obj.offers)

            for entity in config.entities:
                rate = calculate_concession_rate(flat_history, entity.name)
                console.print(f"  {entity.name} concession rate: {rate:.3f} per round")

    else:
        console.print(f"\n[red]Impasse Reason:[/red] {outcome.impasse_reason}")
        console.print("\n[yellow]Last Known Positions:[/yellow]")
        for entity, utility in outcome.final_utilities.items():
            console.print(f"  {entity}: {utility:.3f}")

    # Option to run batch analysis
    console.print("\n[yellow]Would you like to run batch analysis? (y/n):[/yellow]")
    if input().strip().lower() == 'y':
        run_batch_demo(config)


def run_batch_demo(config: SimulationConfig):
    """Run batch analysis demonstration."""

    console.print("\n[cyan]Running 100 simulations with parameter variations...[/cyan]")

    runner = BatchNegotiationRunner(config)
    results = runner.run_batch(100)

    analysis = runner.analyze_results()

    console.print("\n[bold cyan]Batch Analysis Results:[/bold cyan]")
    console.print(f"Success Rate: {analysis['success_rate']:.1%}")
    console.print(f"Average Rounds: {analysis['average_rounds']:.1f}")
    console.print(f"Pareto Optimal Rate: {analysis['pareto_optimal_rate']:.1%}")

    if analysis.get('average_utilities'):
        console.print("\n[cyan]Average Utilities (successful deals):[/cyan]")
        for entity, utility in analysis['average_utilities'].items():
            percentage = utility * 100
            bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
            console.print(f"  {entity:20} {bar} {percentage:.1f}%")


def quick_start():
    """Quick start function for testing."""
    config = create_simple_negotiation()
    engine = NegotiationEngine(config)
    outcome = engine.run()
    return outcome


if __name__ == "__main__":
    # Check if running in interactive mode or with arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            # Quick test mode
            outcome = quick_start()
            console.print(outcome.summary())
        elif sys.argv[1] == "cli":
            # Run CLI
            from cli import app
            app()
    else:
        # Run interactive demo
        run_demo()
