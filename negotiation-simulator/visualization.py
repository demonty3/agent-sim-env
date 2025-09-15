"""
Advanced visualization module for negotiation analysis.
Creates charts, heatmaps, and interactive visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from pathlib import Path

from models import (
    Entity, Issue, NegotiationOutcome, SimulationConfig,
    Offer, NegotiationRound
)
from protocol import BatchNegotiationRunner
from utilities import find_pareto_frontier, calculate_joint_utility


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ===== OUTCOME VISUALIZATIONS =====

def plot_utility_progression(outcome: NegotiationOutcome,
                           save_path: Optional[Path] = None,
                           show: bool = True) -> plt.Figure:
    """
    Plot how utilities evolved during negotiation.
    """
    if not outcome.transcript:
        return None

    # Extract data
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
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot lines for each entity
    for entity, utilities in utilities_by_entity.items():
        ax.plot(rounds[:len(utilities)], utilities,
               marker='o', label=entity, linewidth=2, markersize=6)

    # Add acceptance threshold zones
    ax.axhspan(0.0, 0.5, alpha=0.1, color='red', label='Unacceptable')
    ax.axhspan(0.5, 0.7, alpha=0.1, color='yellow', label='Marginal')
    ax.axhspan(0.7, 1.0, alpha=0.1, color='green', label='Good')

    # Mark final outcome
    if outcome.success:
        ax.axvline(x=outcome.rounds_taken, color='green',
                  linestyle='--', alpha=0.7, label='Agreement')
    else:
        ax.axvline(x=outcome.rounds_taken, color='red',
                  linestyle='--', alpha=0.7, label='Impasse')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Utility', fontsize=12)
    ax.set_title('Negotiation Utility Progression', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_offer_space(outcome: NegotiationOutcome,
                    issue_x: str,
                    issue_y: str,
                    save_path: Optional[Path] = None,
                    show: bool = True) -> plt.Figure:
    """
    Plot offers in 2D issue space showing negotiation trajectory.
    """
    if not outcome.transcript:
        return None

    # Extract offer data
    offers_data = []
    for round_obj in outcome.transcript:
        for offer in round_obj.offers:
            if issue_x in offer.values and issue_y in offer.values:
                offers_data.append({
                    'round': round_obj.round_num,
                    'proposer': offer.proposer,
                    issue_x: offer.values[issue_x],
                    issue_y: offer.values[issue_y],
                    'status': offer.status.value
                })

    if not offers_data:
        return None

    df = pd.DataFrame(offers_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory for each proposer
    for proposer in df['proposer'].unique():
        proposer_df = df[df['proposer'] == proposer].sort_values('round')

        # Plot line
        ax.plot(proposer_df[issue_x], proposer_df[issue_y],
               'o-', label=proposer, alpha=0.7, markersize=8)

        # Add round numbers
        for idx, row in proposer_df.iterrows():
            ax.annotate(str(row['round']),
                       (row[issue_x], row[issue_y]),
                       fontsize=8, ha='center', va='center')

    # Mark final agreement if exists
    if outcome.success and outcome.final_agreement:
        if issue_x in outcome.final_agreement and issue_y in outcome.final_agreement:
            ax.plot(outcome.final_agreement[issue_x],
                   outcome.final_agreement[issue_y],
                   '*', color='gold', markersize=20,
                   label='Final Agreement', zorder=5)

    ax.set_xlabel(issue_x, fontsize=12)
    ax.set_ylabel(issue_y, fontsize=12)
    ax.set_title(f'Negotiation Trajectory in {issue_x}-{issue_y} Space',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_pareto_frontier(entities: List[Entity],
                        issues: List[Issue],
                        samples: int = 1000,
                        highlight_outcome: Optional[NegotiationOutcome] = None,
                        save_path: Optional[Path] = None,
                        show: bool = True) -> plt.Figure:
    """
    Plot Pareto frontier for 2-entity negotiations.
    """
    if len(entities) != 2:
        print("Pareto frontier plot only supports 2-entity negotiations")
        return None

    # Generate random offers
    offers = []
    utilities = []

    for _ in range(samples):
        offer = {}
        for issue in issues:
            if issue.divisible:
                value = np.random.uniform(issue.min_value, issue.max_value)
            else:
                value = np.random.choice([issue.min_value, issue.max_value])
            offer[issue.name] = value

        offers.append(offer)
        utilities.append([
            entities[0].utility_function.calculate_utility(offer),
            entities[1].utility_function.calculate_utility(offer)
        ])

    utilities = np.array(utilities)

    # Find Pareto frontier
    frontier_offers = find_pareto_frontier(offers, entities)
    frontier_utilities = np.array([
        [entities[0].utility_function.calculate_utility(o),
         entities[1].utility_function.calculate_utility(o)]
        for o in frontier_offers
    ])

    # Sort frontier for plotting
    if len(frontier_utilities) > 0:
        sorted_indices = np.argsort(frontier_utilities[:, 0])
        frontier_utilities = frontier_utilities[sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all offers
    ax.scatter(utilities[:, 0], utilities[:, 1],
              alpha=0.3, s=20, label='Possible Outcomes')

    # Plot Pareto frontier
    if len(frontier_utilities) > 0:
        ax.plot(frontier_utilities[:, 0], frontier_utilities[:, 1],
               'r-', linewidth=2, label='Pareto Frontier')
        ax.scatter(frontier_utilities[:, 0], frontier_utilities[:, 1],
                  color='red', s=50, zorder=5)

    # Add reservation point
    reservation_utilities = [
        entities[0].min_acceptable_utility,
        entities[1].min_acceptable_utility
    ]
    ax.plot(reservation_utilities[0], reservation_utilities[1],
           'ks', markersize=10, label='Reservation Point')

    # Add ZOPA rectangle
    ax.add_patch(patches.Rectangle(
        (reservation_utilities[0], reservation_utilities[1]),
        1 - reservation_utilities[0],
        1 - reservation_utilities[1],
        linewidth=2, edgecolor='green', facecolor='green', alpha=0.1,
        label='ZOPA'
    ))

    # Highlight actual outcome if provided
    if highlight_outcome and highlight_outcome.success:
        outcome_utils = [
            highlight_outcome.final_utilities.get(entities[0].name, 0),
            highlight_outcome.final_utilities.get(entities[1].name, 0)
        ]
        ax.plot(outcome_utils[0], outcome_utils[1],
               '*', color='gold', markersize=20,
               label='Actual Outcome', zorder=10)

    ax.set_xlabel(f'{entities[0].name} Utility', fontsize=12)
    ax.set_ylabel(f'{entities[1].name} Utility', fontsize=12)
    ax.set_title('Pareto Frontier Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


# ===== BATCH ANALYSIS VISUALIZATIONS =====

def plot_batch_success_analysis(results: List[NegotiationOutcome],
                               save_path: Optional[Path] = None,
                               show: bool = True) -> plt.Figure:
    """
    Analyze batch results with multiple subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Success rate over time
    ax = axes[0, 0]
    success_cumulative = []
    for i, outcome in enumerate(results):
        if i == 0:
            success_cumulative.append(1 if outcome.success else 0)
        else:
            prev = success_cumulative[-1]
            success_cumulative.append(
                (prev * i + (1 if outcome.success else 0)) / (i + 1)
            )

    ax.plot(range(len(results)), success_cumulative, linewidth=2)
    ax.set_xlabel('Simulation #')
    ax.set_ylabel('Cumulative Success Rate')
    ax.set_title('Success Rate Convergence')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 2. Rounds distribution
    ax = axes[0, 1]
    rounds_data = [o.rounds_taken for o in results]
    ax.hist(rounds_data, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rounds_data), color='red',
              linestyle='--', linewidth=2, label=f'Mean: {np.mean(rounds_data):.1f}')
    ax.set_xlabel('Rounds to Completion')
    ax.set_ylabel('Frequency')
    ax.set_title('Rounds Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Utility distribution (successful only)
    ax = axes[1, 0]
    successful = [o for o in results if o.success]
    if successful:
        all_utilities = []
        for outcome in successful:
            all_utilities.extend(outcome.final_utilities.values())

        ax.hist(all_utilities, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_utilities), color='red',
                  linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_utilities):.2f}')
        ax.set_xlabel('Final Utility')
        ax.set_ylabel('Frequency')
        ax.set_title('Utility Distribution (Successful Negotiations)')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Impasse reasons
    ax = axes[1, 1]
    impasse_reasons = {}
    for outcome in results:
        if not outcome.success and outcome.impasse_reason:
            reason = outcome.impasse_reason.split(':')[0]  # Simplify reason
            impasse_reasons[reason] = impasse_reasons.get(reason, 0) + 1

    if impasse_reasons:
        reasons = list(impasse_reasons.keys())
        counts = list(impasse_reasons.values())
        ax.barh(reasons, counts)
        ax.set_xlabel('Count')
        ax.set_title('Impasse Reasons')
    else:
        ax.text(0.5, 0.5, 'No Impasses', ha='center', va='center', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Batch Analysis - {len(results)} Simulations',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_parameter_sensitivity(base_config: SimulationConfig,
                              param_path: str,
                              param_range: Tuple[float, float],
                              n_samples: int = 20,
                              n_runs_per_sample: int = 10,
                              save_path: Optional[Path] = None,
                              show: bool = True) -> plt.Figure:
    """
    Analyze sensitivity to a specific parameter.
    """
    import copy

    param_values = np.linspace(param_range[0], param_range[1], n_samples)
    success_rates = []
    avg_utilities = []
    avg_rounds = []

    for param_value in param_values:
        # Create modified config
        config = copy.deepcopy(base_config)

        # Apply parameter (simplified - would need proper path parsing)
        # This is a placeholder - real implementation would parse the path
        for entity in config.entities:
            if 'accept_threshold' in param_path:
                entity.policy.params.accept_threshold = param_value
            elif 'concession_rate' in param_path:
                entity.policy.params.concession_rate = param_value

        # Run batch
        runner = BatchNegotiationRunner(config)
        results = runner.run_batch(n_runs_per_sample)
        analysis = runner.analyze_results()

        success_rates.append(analysis['success_rate'])
        avg_rounds.append(analysis['average_rounds'])

        # Calculate average utility
        successful = [r for r in results if r.success]
        if successful:
            all_utils = []
            for r in successful:
                all_utils.extend(r.final_utilities.values())
            avg_utilities.append(np.mean(all_utils) if all_utils else 0)
        else:
            avg_utilities.append(0)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Success rate
    axes[0].plot(param_values, success_rates, 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel(param_path)
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate Sensitivity')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Average utility
    axes[1].plot(param_values, avg_utilities, 'o-', linewidth=2, markersize=6)
    axes[1].set_xlabel(param_path)
    axes[1].set_ylabel('Average Utility')
    axes[1].set_title('Utility Sensitivity')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Average rounds
    axes[2].plot(param_values, avg_rounds, 'o-', linewidth=2, markersize=6)
    axes[2].set_xlabel(param_path)
    axes[2].set_ylabel('Average Rounds')
    axes[2].set_title('Negotiation Length Sensitivity')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Parameter Sensitivity Analysis: {param_path}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


# ===== HEATMAPS AND MATRICES =====

def plot_utility_heatmap(entities: List[Entity],
                        issues: List[Issue],
                        resolution: int = 20,
                        save_path: Optional[Path] = None,
                        show: bool = True) -> plt.Figure:
    """
    Create heatmap showing utility landscape for 2 issues.
    """
    if len(issues) < 2:
        print("Need at least 2 issues for heatmap")
        return None

    # Use first two issues
    issue_x, issue_y = issues[0], issues[1]

    # Create grid
    x_range = np.linspace(issue_x.min_value, issue_x.max_value, resolution)
    y_range = np.linspace(issue_y.min_value, issue_y.max_value, resolution)

    # Calculate utilities for each entity
    n_entities = len(entities)
    fig, axes = plt.subplots(1, n_entities, figsize=(6 * n_entities, 5))

    if n_entities == 1:
        axes = [axes]

    for idx, entity in enumerate(entities):
        utility_grid = np.zeros((resolution, resolution))

        for i, y_val in enumerate(y_range):
            for j, x_val in enumerate(x_range):
                offer = {
                    issue_x.name: x_val,
                    issue_y.name: y_val
                }
                # Add other issues at midpoint
                for issue in issues[2:]:
                    offer[issue.name] = (issue.min_value + issue.max_value) / 2

                utility_grid[i, j] = entity.utility_function.calculate_utility(offer)

        # Plot heatmap
        im = axes[idx].imshow(utility_grid, extent=[issue_x.min_value, issue_x.max_value,
                                                     issue_y.min_value, issue_y.max_value],
                              origin='lower', cmap='viridis', aspect='auto')

        axes[idx].set_xlabel(issue_x.name)
        axes[idx].set_ylabel(issue_y.name)
        axes[idx].set_title(f'{entity.name} Utility')

        # Add colorbar
        plt.colorbar(im, ax=axes[idx])

        # Mark ideal point
        if issue_x.name in entity.utility_function.ideal_values:
            ideal_x = entity.utility_function.ideal_values[issue_x.name]
            ideal_y = entity.utility_function.ideal_values.get(issue_y.name,
                                                              (issue_y.min_value + issue_y.max_value) / 2)
            axes[idx].plot(ideal_x, ideal_y, 'r*', markersize=15, label='Ideal')
            axes[idx].legend()

    plt.suptitle('Utility Landscapes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_concession_patterns(outcome: NegotiationOutcome,
                            save_path: Optional[Path] = None,
                            show: bool = True) -> plt.Figure:
    """
    Visualize concession patterns for each entity.
    """
    if not outcome.transcript or len(outcome.transcript) < 3:
        return None

    # Extract concession data
    entities_data = {}

    for round_obj in outcome.transcript:
        for offer in round_obj.offers:
            proposer = offer.proposer
            if proposer not in entities_data:
                entities_data[proposer] = {
                    'rounds': [],
                    'utilities': [],
                    'concessions': []
                }

            entities_data[proposer]['rounds'].append(round_obj.round_num)
            utility = offer.utility_scores.get(proposer, 0)
            entities_data[proposer]['utilities'].append(utility)

            if len(entities_data[proposer]['utilities']) > 1:
                concession = entities_data[proposer]['utilities'][-2] - utility
                entities_data[proposer]['concessions'].append(concession)

    # Create plot
    n_entities = len(entities_data)
    fig, axes = plt.subplots(n_entities, 2, figsize=(12, 4 * n_entities))

    if n_entities == 1:
        axes = axes.reshape(1, -1)

    for idx, (entity, data) in enumerate(entities_data.items()):
        # Utility over time
        axes[idx, 0].plot(data['rounds'], data['utilities'],
                         'o-', linewidth=2, markersize=6)
        axes[idx, 0].set_ylabel('Utility')
        axes[idx, 0].set_title(f'{entity} - Utility Progression')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_ylim([0, 1])

        # Concession amounts
        if data['concessions']:
            rounds_for_concessions = data['rounds'][1:]
            axes[idx, 1].bar(rounds_for_concessions, data['concessions'],
                           alpha=0.7, edgecolor='black')
            axes[idx, 1].set_ylabel('Concession Amount')
            axes[idx, 1].set_title(f'{entity} - Concession Pattern')
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].axhline(y=0, color='black', linewidth=1)

        if idx == n_entities - 1:
            axes[idx, 0].set_xlabel('Round')
            axes[idx, 1].set_xlabel('Round')

    plt.suptitle('Concession Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig


# ===== ANIMATED VISUALIZATIONS =====

def create_negotiation_animation(outcome: NegotiationOutcome,
                                issue_x: str,
                                issue_y: str,
                                save_path: Optional[Path] = None) -> FuncAnimation:
    """
    Create animated visualization of negotiation progression.
    """
    if not outcome.transcript:
        return None

    # Prepare data
    offers_by_round = []
    for round_obj in outcome.transcript:
        round_offers = []
        for offer in round_obj.offers:
            if issue_x in offer.values and issue_y in offer.values:
                round_offers.append({
                    'proposer': offer.proposer,
                    'x': offer.values[issue_x],
                    'y': offer.values[issue_y],
                    'utility': offer.utility_scores
                })
        if round_offers:
            offers_by_round.append(round_offers)

    if not offers_by_round:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set limits
    x_vals = [o['x'] for round_offers in offers_by_round for o in round_offers]
    y_vals = [o['y'] for round_offers in offers_by_round for o in round_offers]
    ax.set_xlim(min(x_vals) * 0.9, max(x_vals) * 1.1)
    ax.set_ylim(min(y_vals) * 0.9, max(y_vals) * 1.1)

    ax.set_xlabel(issue_x)
    ax.set_ylabel(issue_y)
    ax.set_title('Negotiation Animation')

    # Initialize plot elements
    lines = {}
    points = {}

    def init():
        return []

    def animate(frame):
        ax.clear()
        ax.set_xlim(min(x_vals) * 0.9, max(x_vals) * 1.1)
        ax.set_ylim(min(y_vals) * 0.9, max(y_vals) * 1.1)
        ax.set_xlabel(issue_x)
        ax.set_ylabel(issue_y)
        ax.set_title(f'Negotiation Animation - Round {frame + 1}')

        # Plot history up to current frame
        proposer_history = {}
        for round_idx in range(frame + 1):
            for offer in offers_by_round[round_idx]:
                proposer = offer['proposer']
                if proposer not in proposer_history:
                    proposer_history[proposer] = {'x': [], 'y': []}
                proposer_history[proposer]['x'].append(offer['x'])
                proposer_history[proposer]['y'].append(offer['y'])

        # Plot trajectories
        for proposer, history in proposer_history.items():
            ax.plot(history['x'], history['y'], 'o-', label=proposer, alpha=0.7)

        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax.get_children()

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(offers_by_round), interval=500,
                        blit=False, repeat=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=2)

    return anim


# ===== UTILITY FUNCTIONS =====

def generate_negotiation_report(outcome: NegotiationOutcome,
                               config: SimulationConfig,
                               output_dir: Path):
    """
    Generate comprehensive PDF report with all visualizations.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate all plots
    plot_utility_progression(outcome, save_path=output_dir / "utility_progression.png", show=False)

    if len(config.issues) >= 2:
        plot_offer_space(outcome,
                        config.issues[0].name,
                        config.issues[1].name,
                        save_path=output_dir / "offer_space.png",
                        show=False)

    if len(config.entities) == 2:
        plot_pareto_frontier(config.entities,
                           config.issues,
                           highlight_outcome=outcome,
                           save_path=output_dir / "pareto_frontier.png",
                           show=False)

    plot_concession_patterns(outcome,
                           save_path=output_dir / "concession_patterns.png",
                           show=False)

    print(f"Report generated in {output_dir}")
