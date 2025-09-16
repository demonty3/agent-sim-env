"""
Command-line interface for the negotiation simulator.
Provides commands to run simulations, analyze results, and visualize outcomes.
"""

import typer
import yaml
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import numpy as np

from models import (
    Entity, Issue, UtilityFunction, NegotiationPolicy,
    PolicyType, PolicyParameters, SimulationConfig
)
from protocol import NegotiationEngine, BatchNegotiationRunner
from utilities import analyze_negotiation_space, find_nash_bargaining_solution

app = typer.Typer(help="AI Agent Negotiation Simulator")
console = Console()


# ===== COMMANDS =====

@app.command()
def run(
    config_file: Path = typer.Argument(..., help="Path to YAML configuration file"),
    output: Optional[Path] = typer.Option(None, help="Output file for results"),
    visualize: bool = typer.Option(False, "--viz", help="Show visualization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run a single negotiation simulation."""

    # Load configuration
    console.print(f"[cyan]Loading configuration from {config_file}...[/cyan]")
    config = load_config(config_file)

    # Run negotiation
    console.print("[yellow]Starting negotiation...[/yellow]")
    engine = NegotiationEngine(config)
    outcome = engine.run()

    # Display results
    display_outcome(outcome, verbose)

    # Save results if requested
    if output:
        save_outcome(outcome, output)
        console.print(f"[green]Results saved to {output}[/green]")

    # Visualize if requested
    if visualize:
        visualize_negotiation(outcome)


@app.command()
def batch(
    config_file: Path = typer.Argument(..., help="Path to YAML configuration file"),
    n_runs: int = typer.Option(100, help="Number of simulations to run"),
    output: Optional[Path] = typer.Option(None, help="Output file for results"),
    vary: Optional[str] = typer.Option(None, help="Parameters to vary (JSON format)")
):
    """Run batch simulations with parameter variations."""

    # Load configuration
    config = load_config(config_file)

    # Parse variation parameters
    vary_params = None
    if vary:
        vary_params = json.loads(vary)

    # Run batch
    console.print(f"[yellow]Running {n_runs} simulations...[/yellow]")
    runner = BatchNegotiationRunner(config)

    results = []
    for _ in track(range(n_runs), description="Running simulations..."):
        results.append(runner.run_batch(1, vary_params)[0])

    runner.results = results

    # Analyze results
    analysis = runner.analyze_results()
    display_batch_analysis(analysis)

    # Save if requested
    if output:
        save_batch_results(results, analysis, output)
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def analyze(
    config_file: Path = typer.Argument(..., help="Path to YAML configuration file"),
    samples: int = typer.Option(1000, help="Number of samples for analysis")
):
    """Analyze the negotiation space for a given configuration."""

    # Load configuration
    config = load_config(config_file)

    console.print(f"[yellow]Analyzing negotiation space with {samples} samples...[/yellow]")

    # Perform analysis
    analysis = analyze_negotiation_space(
        config.entities,
        config.issues,
        samples=samples
    )

    # Display analysis
    display_space_analysis(analysis)

    # Find Nash solution
    if analysis['has_zopa']:
        nash = find_nash_bargaining_solution(config.entities, config.issues, samples)
        console.print("\n[cyan]Nash Bargaining Solution:[/cyan]")
        for issue, value in nash.items():
            console.print(f"  {issue}: {value:.2f}")


@app.command()
def example():
    """Generate example configuration files."""

    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    # Generate trade negotiation example
    trade_config = create_trade_example()
    trade_path = examples_dir / "trade_negotiation.yaml"
    with open(trade_path, 'w') as f:
        yaml.dump(trade_config, f, default_flow_style=False)

    # Generate resource allocation example
    resource_config = create_resource_example()
    resource_path = examples_dir / "resource_allocation.yaml"
    with open(resource_path, 'w') as f:
        yaml.dump(resource_config, f, default_flow_style=False)

    console.print(f"[green]Example configurations created in {examples_dir}/[/green]")
    console.print(f"  - {trade_path}")
    console.print(f"  - {resource_path}")


# ===== HELPER FUNCTIONS =====

def load_config(config_file: Path) -> SimulationConfig:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)

    # Parse entities
    entities = []
    for entity_data in data['entities']:
        # Create utility function
        utility_fn = UtilityFunction(
            weights=entity_data['utility']['weights'],
            ideal_values=entity_data['utility']['ideal_values'],
            reservation_values=entity_data['utility']['reservation_values']
        )

        # Create policy
        policy_params = PolicyParameters(**entity_data['policy'].get('params', {}))
        policy = NegotiationPolicy(
            type=PolicyType(entity_data['policy']['type']),
            params=policy_params
        )

        # Create entity
        entity = Entity(
            name=entity_data['name'],
            type=entity_data.get('type', 'country'),
            utility_function=utility_fn,
            policy=policy,
            max_rounds=entity_data.get('max_rounds', 100),
            min_acceptable_utility=entity_data.get('min_acceptable_utility', 0.5)
        )
        entities.append(entity)

    # Parse issues
    issues = []
    for issue_data in data['issues']:
        issue = Issue(
            name=issue_data['name'],
            min_value=issue_data['min_value'],
            max_value=issue_data['max_value'],
            divisible=issue_data.get('divisible', True),
            unit=issue_data.get('unit')
        )
        issues.append(issue)

    # Create config
    config = SimulationConfig(
        entities=entities,
        issues=issues,
        max_rounds=data.get('max_rounds', 100),
        protocol=data.get('protocol', 'alternating'),
        allow_coalition=data.get('allow_coalition', False),
        track_pareto=data.get('track_pareto', True),
        calculate_nash=data.get('calculate_nash', True)
    )

    return config


def display_outcome(outcome, verbose: bool = False):
    """Display negotiation outcome in a formatted way."""

    # Summary
    console.print(f"\n[bold]{outcome.summary()}[/bold]")

    if outcome.success:
        # Show agreement
        console.print("\n[green]Final Agreement:[/green]")
        for issue, value in outcome.final_agreement.items():
            console.print(f"  {issue}: {value:.2f}")

        # Show utilities
        console.print("\n[cyan]Final Utilities:[/cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Entity")
        table.add_column("Utility", justify="right")

        for entity, utility in outcome.final_utilities.items():
            table.add_row(entity, f"{utility:.3f}")

        console.print(table)

        # Show analysis
        if outcome.pareto_optimal is not None:
            status = "✅ Yes" if outcome.pareto_optimal else "❌ No"
            console.print(f"\nPareto Optimal: {status}")

        if outcome.nash_bargaining_score is not None:
            console.print(f"Nash Product: {outcome.nash_bargaining_score:.3f}")

    # Verbose output shows transcript
    if verbose and outcome.transcript:
        console.print("\n[yellow]Negotiation Transcript:[/yellow]")
        for round_obj in outcome.transcript[-5:]:  # Show last 5 rounds
            console.print(f"\nRound {round_obj.round_num}:")
            for offer in round_obj.offers:
                console.print(f"  Proposer: {offer.proposer}")
                console.print(f"  Status: {offer.status.value}")
                if verbose:
                    console.print(f"  Values: {offer.values}")


def display_batch_analysis(analysis: dict):
    """Display batch simulation analysis."""
    console.print("\n[bold cyan]Batch Analysis Results:[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Success Rate", f"{analysis['success_rate']:.1%}")
    table.add_row("Average Rounds", f"{analysis['average_rounds']:.1f}")
    table.add_row("Pareto Optimal Rate", f"{analysis['pareto_optimal_rate']:.1%}")
    table.add_row("Total Runs", str(analysis['total_runs']))

    console.print(table)

    if analysis.get('average_utilities'):
        console.print("\n[cyan]Average Utilities (successful deals):[/cyan]")
        for entity, utility in analysis['average_utilities'].items():
            console.print(f"  {entity}: {utility:.3f}")


def display_space_analysis(analysis: dict):
    """Display negotiation space analysis."""
    console.print("\n[bold cyan]Negotiation Space Analysis:[/bold cyan]")

    if not analysis['has_zopa']:
        console.print("[red]❌ No Zone of Possible Agreement (ZOPA) exists![/red]")
        console.print("The parties' reservation values are incompatible.")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("ZOPA Size", str(analysis['zopa_size']))
    table.add_row("Pareto Frontier Size", str(analysis['pareto_frontier_size']))
    table.add_row("Nash Product", f"{analysis.get('nash_product', 0):.3f}")
    table.add_row("Max Joint Utility", f"{analysis.get('max_joint_utility', 0):.3f}")
    table.add_row("Min Joint Utility", f"{analysis.get('min_joint_utility', 0):.3f}")

    console.print(table)

    if analysis.get('average_utilities'):
        console.print("\n[cyan]Average Utilities in ZOPA:[/cyan]")
        for entity, utility in analysis['average_utilities'].items():
            console.print(f"  {entity}: {utility:.3f}")


def visualize_negotiation(outcome):
    """Create visualization of negotiation progress."""
    if not outcome.transcript:
        return

    # Extract utility progression
    rounds = []
    utilities_by_entity = {}

    for round_obj in outcome.transcript:
        rounds.append(round_obj.round_num)
        for offer in round_obj.offers:
            for entity, utility in offer.utility_scores.items():
                if entity not in utilities_by_entity:
                    utilities_by_entity[entity] = []
                utilities_by_entity[entity].append(utility)

    # Create plot
    plt.figure(figsize=(10, 6))

    for entity, utilities in utilities_by_entity.items():
        plt.plot(rounds[:len(utilities)], utilities, label=entity, marker='o')

    plt.xlabel('Round')
    plt.ylabel('Utility')
    plt.title('Negotiation Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if outcome.success:
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Min Acceptable')

    plt.tight_layout()
    plt.show()


def save_outcome(outcome, output_path: Path):
    """Save negotiation outcome to file."""
    data = {
        'success': outcome.success,
        'rounds_taken': outcome.rounds_taken,
        'final_agreement': outcome.final_agreement,
        'final_utilities': outcome.final_utilities,
        'impasse_reason': outcome.impasse_reason,
        'pareto_optimal': outcome.pareto_optimal,
        'nash_bargaining_score': outcome.nash_bargaining_score
    }

    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


def save_batch_results(results, analysis, output_path: Path):
    """Save batch results to file."""
    data = {
        'analysis': analysis,
        'outcomes': [
            {
                'success': r.success,
                'rounds': r.rounds_taken,
                'utilities': r.final_utilities,
                'agreement': r.final_agreement
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# ===== EXAMPLE CONFIGURATIONS =====

def create_trade_example() -> dict:
    """Create example trade negotiation configuration."""
    return {
        'entities': [
            {
                'name': 'Country_A',
                'type': 'country',
                'utility': {
                    'weights': {'price': 0.7, 'quantity': 0.3},
                    'ideal_values': {'price': 100, 'quantity': 1000},
                    'reservation_values': {'price': 80, 'quantity': 500}
                },
                'policy': {
                    'type': 'linear_concession',
                    'params': {
                        'accept_threshold': 0.7,
                        'concession_rate': 0.05,
                        'patience': 20
                    }
                }
            },
            {
                'name': 'Country_B',
                'type': 'country',
                'utility': {
                    'weights': {'price': 0.5, 'quantity': 0.5},
                    'ideal_values': {'price': 70, 'quantity': 1500},
                    'reservation_values': {'price': 90, 'quantity': 800}
                },
                'policy': {
                    'type': 'tit_for_tat',
                    'params': {
                        'accept_threshold': 0.65,
                        'stubbornness': 0.3
                    }
                }
            }
        ],
        'issues': [
            {
                'name': 'price',
                'min_value': 50,
                'max_value': 150,
                'unit': 'USD'
            },
            {
                'name': 'quantity',
                'min_value': 100,
                'max_value': 2000,
                'unit': 'units'
            }
        ],
        'max_rounds': 50,
        'protocol': 'alternating'
    }


def create_resource_example() -> dict:
    """Create example resource allocation configuration."""
    return {
        'entities': [
            {
                'name': 'Aquila',
                'type': 'country',
                'utility': {
                    'weights': {'water': 0.6, 'energy': 0.4},
                    'ideal_values': {'water': 80, 'energy': 60},
                    'reservation_values': {'water': 40, 'energy': 30}
                },
                'policy': {
                    'type': 'boulware',
                    'params': {
                        'accept_threshold': 0.75,
                        'initial_demand': 0.95,
                        'patience': 30
                    }
                }
            },
            {
                'name': 'Beringia',
                'type': 'country',
                'utility': {
                    'weights': {'water': 0.4, 'energy': 0.6},
                    'ideal_values': {'water': 60, 'energy': 80},
                    'reservation_values': {'water': 30, 'energy': 40}
                },
                'policy': {
                    'type': 'conceder',
                    'params': {
                        'accept_threshold': 0.6,
                        'concession_rate': 0.1
                    }
                }
            },
            {
                'name': 'Cascadia',
                'type': 'country',
                'utility': {
                    'weights': {'water': 0.5, 'energy': 0.5},
                    'ideal_values': {'water': 70, 'energy': 70},
                    'reservation_values': {'water': 35, 'energy': 35}
                },
                'policy': {
                    'type': 'adaptive',
                    'params': {
                        'accept_threshold': 0.65,
                        'learning_rate': 0.2
                    }
                }
            }
        ],
        'issues': [
            {
                'name': 'water',
                'min_value': 0,
                'max_value': 100,
                'unit': 'million_gallons'
            },
            {
                'name': 'energy',
                'min_value': 0,
                'max_value': 100,
                'unit': 'GWh'
            }
        ],
        'max_rounds': 100,
        'protocol': 'simultaneous'
    }


if __name__ == "__main__":
    app()
