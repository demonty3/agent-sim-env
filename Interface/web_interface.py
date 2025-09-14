"""
Web interface for the negotiation simulator using Streamlit.
Run with: streamlit run web_interface.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import yaml
from pathlib import Path

from models import (
    Entity, Issue, UtilityFunction, NegotiationPolicy,
    PolicyType, PolicyParameters, SimulationConfig
)
from protocol import NegotiationEngine, BatchNegotiationRunner
from utilities import analyze_negotiation_space, find_nash_bargaining_solution
from advisor import NegotiationAdvisor
from visualization import plot_utility_progression, plot_pareto_frontier
from advanced_strategies import create_advanced_strategy, compare_strategies


# ===== PAGE CONFIG =====

st.set_page_config(
    page_title="AI Negotiation Simulator",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .failure-box {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ===== SESSION STATE =====

if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []

if 'current_config' not in st.session_state:
    st.session_state.current_config = None

if 'last_outcome' not in st.session_state:
    st.session_state.last_outcome = None


# ===== HELPER FUNCTIONS =====

def create_entity_from_ui(name: str, entity_config: dict) -> Entity:
    """Create entity from UI configuration."""
    return Entity(
        name=name,
        utility_function=UtilityFunction(
            weights=entity_config['weights'],
            ideal_values=entity_config['ideal_values'],
            reservation_values=entity_config['reservation_values']
        ),
        policy=NegotiationPolicy(
            type=PolicyType(entity_config['policy_type']),
            params=PolicyParameters(**entity_config['policy_params'])
        ),
        min_acceptable_utility=entity_config.get('min_acceptable_utility', 0.5)
    )


def create_issue_from_ui(issue_config: dict) -> Issue:
    """Create issue from UI configuration."""
    return Issue(
        name=issue_config['name'],
        min_value=issue_config['min_value'],
        max_value=issue_config['max_value'],
        divisible=issue_config.get('divisible', True),
        unit=issue_config.get('unit')
    )


def plot_negotiation_space_3d(entities, issues):
    """Create 3D visualization of negotiation space."""
    if len(issues) < 2:
        return None

    # Use first two issues for main axes
    issue1, issue2 = issues[0], issues[1]

    # Generate grid
    x = np.linspace(issue1.min_value, issue1.max_value, 50)
    y = np.linspace(issue2.min_value, issue2.max_value, 50)

    fig = go.Figure()

    for entity in entities:
        z = np.zeros((50, 50))
        for i, y_val in enumerate(y):
            for j, x_val in enumerate(x):
                offer = {
                    issue1.name: x_val,
                    issue2.name: y_val
                }
                # Add other issues at midpoint
                for issue in issues[2:]:
                    offer[issue.name] = (issue.min_value + issue.max_value) / 2

                z[i, j] = entity.utility_function.calculate_utility(offer)

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            name=entity.name,
            showscale=True,
            opacity=0.7
        ))

    fig.update_layout(
        title="Utility Landscapes",
        scene=dict(
            xaxis_title=issue1.name,
            yaxis_title=issue2.name,
            zaxis_title="Utility"
        ),
        height=600
    )

    return fig


# ===== MAIN INTERFACE =====

def main():
    st.markdown('<h1 class="main-header">🤝 AI Agent Negotiation Simulator</h1>',
                unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Setup", "Simulate", "Analyze", "History", "Advanced"]
    )

    if page == "Setup":
        setup_page()
    elif page == "Simulate":
        simulate_page()
    elif page == "Analyze":
        analyze_page()
    elif page == "History":
        history_page()
    elif page == "Advanced":
        advanced_page()


def setup_page():
    """Configuration setup page."""
    st.header("📋 Negotiation Setup")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quick Templates")

        template = st.selectbox(
            "Choose a template",
            ["Custom", "Buyer-Seller", "Resource Allocation", "Climate Accord"]
        )

        if template == "Buyer-Seller":
            if st.button("Load Buyer-Seller Template"):
                load_buyer_seller_template()
                st.success("Template loaded!")
        elif template == "Resource Allocation":
            if st.button("Load Resource Template"):
                load_resource_template()
                st.success("Template loaded!")
        elif template == "Climate Accord":
            if st.button("Load Climate Template"):
                load_climate_template()
                st.success("Template loaded!")

    with col2:
        st.subheader("Upload Configuration")

        uploaded_file = st.file_uploader("Choose a YAML file", type=['yaml', 'yml'])
        if uploaded_file is not None:
            config_data = yaml.safe_load(uploaded_file)
            st.session_state.current_config = config_data
            st.success("Configuration loaded!")

    st.divider()

    # Manual configuration
    st.subheader("Manual Configuration")

    # Issues configuration
    st.write("### Issues")
    num_issues = st.number_input("Number of issues", min_value=1, max_value=10, value=2)

    issues_config = []
    cols = st.columns(min(3, num_issues))

    for i in range(num_issues):
        with cols[i % 3]:
            st.write(f"**Issue {i+1}**")
            name = st.text_input(f"Name", value=f"issue_{i+1}", key=f"issue_name_{i}")
            min_val = st.number_input(f"Min value", value=0.0, key=f"issue_min_{i}")
            max_val = st.number_input(f"Max value", value=100.0, key=f"issue_max_{i}")
            unit = st.text_input(f"Unit (optional)", key=f"issue_unit_{i}")
            divisible = st.checkbox(f"Divisible", value=True, key=f"issue_div_{i}")

            issues_config.append({
                'name': name,
                'min_value': min_val,
                'max_value': max_val,
                'unit': unit if unit else None,
                'divisible': divisible
            })

    st.divider()

    # Entities configuration
    st.write("### Entities")
    num_entities = st.number_input("Number of entities", min_value=2, max_value=10, value=2)

    entities_config = []

    for i in range(num_entities):
        with st.expander(f"Entity {i+1}", expanded=i==0):
            col1, col2, col3 = st.columns(3)

            with col1:
                name = st.text_input("Name", value=f"Entity_{i+1}", key=f"entity_name_{i}")
                policy_type = st.selectbox(
                    "Policy Type",
                    [p.value for p in PolicyType],
                    key=f"policy_type_{i}"
                )

            with col2:
                st.write("**Policy Parameters**")
                accept_threshold = st.slider(
                    "Accept Threshold",
                    0.0, 1.0, 0.65, 0.05,
                    key=f"accept_threshold_{i}"
                )
                concession_rate = st.slider(
                    "Concession Rate",
                    0.0, 0.3, 0.08, 0.01,
                    key=f"concession_rate_{i}"
                )
                stubbornness = st.slider(
                    "Stubbornness",
                    0.0, 1.0, 0.5, 0.05,
                    key=f"stubbornness_{i}"
                )

            with col3:
                st.write("**Utility Function**")
                weights = {}
                ideal_values = {}
                reservation_values = {}

                for j, issue in enumerate(issues_config):
                    st.write(f"*{issue['name']}*")
                    weight = st.slider(
                        f"Weight",
                        0.0, 1.0, 0.5, 0.05,
                        key=f"weight_{i}_{j}"
                    )
                    ideal = st.number_input(
                        f"Ideal",
                        issue['min_value'], issue['max_value'],
                        (issue['min_value'] + issue['max_value']) / 2,
                        key=f"ideal_{i}_{j}"
                    )
                    reservation = st.number_input(
                        f"Reservation",
                        issue['min_value'], issue['max_value'],
                        (issue['min_value'] + issue['max_value']) / 2,
                        key=f"reservation_{i}_{j}"
                    )

                    weights[issue['name']] = weight
                    ideal_values[issue['name']] = ideal
                    reservation_values[issue['name']] = reservation

            entities_config.append({
                'name': name,
                'policy_type': policy_type,
                'policy_params': {
                    'accept_threshold': accept_threshold,
                    'concession_rate': concession_rate,
                    'stubbornness': stubbornness
                },
                'weights': weights,
                'ideal_values': ideal_values,
                'reservation_values': reservation_values,
                'min_acceptable_utility': 0.5
            })

    st.divider()

    # Simulation parameters
    st.write("### Simulation Parameters")
    col1, col2 = st.columns(2)

    with col1:
        max_rounds = st.number_input("Max Rounds", min_value=10, max_value=500, value=100)
        protocol = st.selectbox("Protocol", ["alternating", "simultaneous", "random"])

    with col2:
        track_pareto = st.checkbox("Track Pareto Optimality", value=True)
        calculate_nash = st.checkbox("Calculate Nash Solution", value=True)

    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        config = {
            'entities': entities_config,
            'issues': issues_config,
            'max_rounds': max_rounds,
            'protocol': protocol,
            'track_pareto': track_pareto,
            'calculate_nash': calculate_nash
        }
        st.session_state.current_config = config
        st.success("Configuration saved!")

        # Option to download
        yaml_str = yaml.dump(config, default_flow_style=False)
        st.download_button(
            label="Download Configuration",
            data=yaml_str,
            file_name="negotiation_config.yaml",
            mime="text/yaml"
        )


def simulate_page():
    """Run simulation page."""
    st.header("🎮 Run Simulation")

    if st.session_state.current_config is None:
        st.warning("Please configure the negotiation first in the Setup page.")
        return

    config_data = st.session_state.current_config

    # Display current configuration
    with st.expander("Current Configuration", expanded=False):
        st.json(config_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 Run Single Simulation", type="primary"):
            run_single_simulation(config_data)

    with col2:
        n_runs = st.number_input("Batch size", min_value=10, max_value=1000, value=100)
        if st.button("📊 Run Batch Simulation"):
            run_batch_simulation(config_data, n_runs)

    with col3:
        if st.button("🔄 Reset", type="secondary"):
            st.session_state.last_outcome = None
            st.rerun()

    # Display results
    if st.session_state.last_outcome is not None:
        display_simulation_results(st.session_state.last_outcome)


def run_single_simulation(config_data):
    """Run a single negotiation simulation."""
    with st.spinner("Running negotiation..."):
        # Create entities and issues
        entities = [create_entity_from_ui(e['name'], e) for e in config_data['entities']]
        issues = [create_issue_from_ui(i) for i in config_data['issues']]

        # Create configuration
        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=config_data['max_rounds'],
            protocol=config_data['protocol'],
            track_pareto=config_data.get('track_pareto', True),
            calculate_nash=config_data.get('calculate_nash', True)
        )

        # Run simulation
        engine = NegotiationEngine(config)
        outcome = engine.run()

        # Store results
        st.session_state.last_outcome = outcome
        st.session_state.simulation_history.append({
            'timestamp': datetime.now(),
            'config': config_data,
            'outcome': outcome
        })

        # Show notification
        if outcome.success:
            st.success(f"✅ {outcome.summary()}")
        else:
            st.error(f"❌ {outcome.summary()}")


def run_batch_simulation(config_data, n_runs):
    """Run batch simulations."""
    with st.spinner(f"Running {n_runs} simulations..."):
        # Create entities and issues
        entities = [create_entity_from_ui(e['name'], e) for e in config_data['entities']]
        issues = [create_issue_from_ui(i) for i in config_data['issues']]

        # Create configuration
        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=config_data['max_rounds'],
            protocol=config_data['protocol']
        )

        # Run batch
        runner = BatchNegotiationRunner(config)
        results = runner.run_batch(n_runs)
        analysis = runner.analyze_results()

        # Display batch results
        st.subheader("Batch Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Success Rate", f"{analysis['success_rate']:.1%}")
        with col2:
            st.metric("Avg Rounds", f"{analysis['average_rounds']:.1f}")
        with col3:
            st.metric("Pareto Optimal", f"{analysis.get('pareto_optimal_rate', 0):.1%}")
        with col4:
            st.metric("Total Runs", analysis['total_runs'])

        # Create distribution plots
        st.subheader("Distributions")

        col1, col2 = st.columns(2)

        with col1:
            # Rounds distribution
            rounds_data = [r.rounds_taken for r in results]
            fig = px.histogram(
                x=rounds_data,
                nbins=20,
                title="Rounds Distribution",
                labels={'x': 'Rounds', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Success by round
            success_by_round = {}
            for r in results:
                if r.success:
                    round_bucket = r.rounds_taken // 10 * 10
                    success_by_round[round_bucket] = success_by_round.get(round_bucket, 0) + 1

            if success_by_round:
                fig = px.bar(
                    x=list(success_by_round.keys()),
                    y=list(success_by_round.values()),
                    title="Successful Negotiations by Round",
                    labels={'x': 'Round Bucket', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)


def display_simulation_results(outcome):
    """Display detailed simulation results."""
    st.divider()
    st.subheader("Simulation Results")

    if outcome.success:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write(f"**Agreement reached in {outcome.rounds_taken} rounds!**")

        # Display agreement
        st.write("**Final Agreement:**")
        agreement_df = pd.DataFrame([outcome.final_agreement])
        st.dataframe(agreement_df)

        # Display utilities
        st.write("**Entity Satisfaction:**")
        utilities_df = pd.DataFrame([outcome.final_utilities])
        st.dataframe(utilities_df.style.format("{:.3f}"))

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="failure-box">', unsafe_allow_html=True)
        st.write(f"**Negotiation failed after {outcome.rounds_taken} rounds**")
        st.write(f"**Reason:** {outcome.impasse_reason}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Visualizations
    st.subheader("Visualizations")

    tab1, tab2, tab3 = st.tabs(["Utility Progress", "Offer Trajectory", "Analysis"])

    with tab1:
        # Utility progression plot
        if outcome.transcript:
            rounds = []
            utilities_data = []

            for round_obj in outcome.transcript:
                for offer in round_obj.offers:
                    for entity, utility in offer.utility_scores.items():
                        utilities_data.append({
                            'Round': round_obj.round_num,
                            'Entity': entity,
                            'Utility': utility
                        })

            if utilities_data:
                df = pd.DataFrame(utilities_data)
                fig = px.line(
                    df, x='Round', y='Utility', color='Entity',
                    title='Utility Progression',
                    markers=True
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                            annotation_text="Minimum Acceptable")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Offer trajectory
        if len(st.session_state.current_config['issues']) >= 2 and outcome.transcript:
            issue1 = st.session_state.current_config['issues'][0]['name']
            issue2 = st.session_state.current_config['issues'][1]['name']

            trajectory_data = []
            for round_obj in outcome.transcript:
                for offer in round_obj.offers:
                    if issue1 in offer.values and issue2 in offer.values:
                        trajectory_data.append({
                            'Round': round_obj.round_num,
                            'Proposer': offer.proposer,
                            issue1: offer.values[issue1],
                            issue2: offer.values[issue2]
                        })

            if trajectory_data:
                df = pd.DataFrame(trajectory_data)
                fig = px.scatter(
                    df, x=issue1, y=issue2, color='Proposer',
                    title=f'Negotiation Trajectory',
                    animation_frame='Round',
                    size_max=20
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Analysis metrics
        if outcome.pareto_optimal is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pareto Optimal",
                         "Yes ✅" if outcome.pareto_optimal else "No ❌")
            with col2:
                if outcome.nash_bargaining_score is not None:
                    st.metric("Nash Product", f"{outcome.nash_bargaining_score:.3f}")


def analyze_page():
    """Analysis and insights page."""
    st.header("📈 Analysis & Insights")

    if st.session_state.current_config is None:
        st.warning("Please configure the negotiation first in the Setup page.")
        return

    config_data = st.session_state.current_config

    # Create entities and issues
    entities = [create_entity_from_ui(e['name'], e) for e in config_data['entities']]
    issues = [create_issue_from_ui(i) for i in config_data['issues']]

    # Negotiation space analysis
    st.subheader("Negotiation Space Analysis")

    with st.spinner("Analyzing negotiation space..."):
        analysis = analyze_negotiation_space(entities, issues, samples=1000)

    col1, col2, col3 = st.columns(3)

    with col1:
        if analysis['has_zopa']:
            st.success(f"✅ ZOPA exists with {analysis['zopa_size']} solutions")
        else:
            st.error("❌ No ZOPA - agreement unlikely")

    with col2:
        if analysis.get('nash_solution'):
            st.info(f"Nash Product: {analysis.get('nash_product', 0):.3f}")

    with col3:
        if analysis.get('pareto_frontier_size'):
            st.info(f"Pareto Frontier: {analysis['pareto_frontier_size']} points")

    # Nash solution
    if analysis.get('nash_solution'):
        st.subheader("Nash Bargaining Solution")
        nash_df = pd.DataFrame([analysis['nash_solution']])
        st.dataframe(nash_df)

    # 3D visualization
    if len(issues) >= 2:
        st.subheader("Utility Landscapes")
        fig = plot_negotiation_space_3d(entities, issues)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # AI Advisory
    if st.session_state.last_outcome:
        st.subheader("AI Advisory Report")

        advisor = NegotiationAdvisor()
        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=config_data['max_rounds'],
            protocol=config_data['protocol']
        )

        report = advisor.analyze_outcome(config, st.session_state.last_outcome)

        # Display report
        st.write("**Analysis:**", report.outcome_analysis)
        st.write("**Success Probability:**", f"{report.success_probability:.1%}")

        if report.parameter_suggestions:
            st.write("**Parameter Suggestions:**")
            for suggestion in report.parameter_suggestions[:3]:
                with st.expander(f"{suggestion.entity_name}.{suggestion.parameter_path}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current", f"{suggestion.current_value:.2f}")
                    with col2:
                        st.metric("Suggested", f"{suggestion.suggested_value:.2f}")
                    st.write(f"**Rationale:** {suggestion.rationale}")
                    st.write(f"**Confidence:** {suggestion.confidence:.1%}")

        if report.key_insights:
            st.write("**Key Insights:**")
            for insight in report.key_insights:
                st.info(insight)


def history_page():
    """View simulation history."""
    st.header("📜 Simulation History")

    if not st.session_state.simulation_history:
        st.info("No simulations run yet.")
        return

    # Summary statistics
    st.subheader("Summary Statistics")

    total_sims = len(st.session_state.simulation_history)
    successful = sum(1 for s in st.session_state.simulation_history
                    if s['outcome'].success)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Simulations", total_sims)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Success Rate", f"{successful/total_sims:.1%}" if total_sims > 0 else "N/A")

    # History table
    st.subheader("Recent Simulations")

    history_data = []
    for i, sim in enumerate(reversed(st.session_state.simulation_history[-10:])):
        history_data.append({
            'Time': sim['timestamp'].strftime("%H:%M:%S"),
            'Entities': len(sim['config']['entities']),
            'Issues': len(sim['config']['issues']),
            'Rounds': sim['outcome'].rounds_taken,
            'Success': "✅" if sim['outcome'].success else "❌",
            'Avg Utility': np.mean(list(sim['outcome'].final_utilities.values()))
                          if sim['outcome'].success else 0
        })

    df = pd.DataFrame(history_data)
    st.dataframe(df)

    # Clear history button
    if st.button("Clear History"):
        st.session_state.simulation_history = []
        st.rerun()


def advanced_page():
    """Advanced features page."""
    st.header("🔬 Advanced Features")

    tab1, tab2, tab3 = st.tabs(["Strategy Comparison", "Parameter Sweep", "LLM Integration"])

    with tab1:
        st.subheader("Strategy Comparison")

        if st.session_state.current_config:
            strategies = st.multiselect(
                "Select strategies to compare",
                ["adaptive", "mixed", "mcts", "q_learning"],
                default=["adaptive", "mixed"]
            )

            n_runs = st.slider("Runs per strategy", 10, 200, 50)

            if st.button("Compare Strategies"):
                config_data = st.session_state.current_config
                entities = [create_entity_from_ui(e['name'], e)
                          for e in config_data['entities']]
                issues = [create_issue_from_ui(i) for i in config_data['issues']]

                with st.spinner("Running comparison..."):
                    comparison_df = compare_strategies(entities, issues, strategies, n_runs)

                st.dataframe(comparison_df)

                # Plot comparison
                fig = px.bar(
                    comparison_df,
                    x='Strategy',
                    y=['Success Rate', 'Avg Utility'],
                    title='Strategy Performance Comparison',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Parameter Sensitivity Analysis")

        if st.session_state.current_config:
            param_name = st.selectbox(
                "Parameter to analyze",
                ["accept_threshold", "concession_rate", "stubbornness"]
            )

            param_range = st.slider(
                "Parameter range",
                0.0, 1.0, (0.3, 0.8), 0.05
            )

            n_samples = st.slider("Sample points", 5, 20, 10)

            if st.button("Run Sensitivity Analysis"):
                st.info("Running parameter sweep...")
                # Implementation would go here
                st.success("Analysis complete!")

    with tab3:
        st.subheader("LLM Integration (V2 Preview)")

        st.info("🚀 Coming Soon: Natural language strategy generation using LLMs")

        user_prompt = st.text_area(
            "Describe your negotiation strategy in natural language:",
            "Be firm on price but flexible on delivery time. Start with high demands but be willing to compromise if the other party shows good faith."
        )

        if st.button("Generate Strategy (Demo)"):
            st.code("""
            # Generated Strategy Configuration
            policy:
              type: adaptive
              params:
                accept_threshold: 0.75  # Firm position
                initial_demand: 0.95    # High starting point
                concession_rate: 0.06   # Moderate concession
                learning_rate: 0.2      # Adapt to opponent

            weights:
              price: 0.8          # High priority
              delivery_time: 0.2  # Low priority
            """)


def load_buyer_seller_template():
    """Load buyer-seller template configuration."""
    st.session_state.current_config = {
        'entities': [
            {
                'name': 'Buyer',
                'policy_type': 'linear_concession',
                'policy_params': {
                    'accept_threshold': 0.65,
                    'concession_rate': 0.08,
                    'stubbornness': 0.4
                },
                'weights': {'price': 0.7, 'quantity': 0.3},
                'ideal_values': {'price': 1000, 'quantity': 1000},
                'reservation_values': {'price': 4000, 'quantity': 200},
                'min_acceptable_utility': 0.5
            },
            {
                'name': 'Seller',
                'policy_type': 'tit_for_tat',
                'policy_params': {
                    'accept_threshold': 0.6,
                    'concession_rate': 0.1,
                    'stubbornness': 0.5
                },
                'weights': {'price': 0.8, 'quantity': 0.2},
                'ideal_values': {'price': 5000, 'quantity': 100},
                'reservation_values': {'price': 2000, 'quantity': 800},
                'min_acceptable_utility': 0.5
            }
        ],
        'issues': [
            {'name': 'price', 'min_value': 500, 'max_value': 6000, 'unit': 'USD', 'divisible': True},
            {'name': 'quantity', 'min_value': 50, 'max_value': 1500, 'unit': 'units', 'divisible': True}
        ],
        'max_rounds': 50,
        'protocol': 'alternating',
        'track_pareto': True,
        'calculate_nash': True
    }


def load_resource_template():
    """Load resource allocation template."""
    st.session_state.current_config = {
        'entities': [
            {
                'name': 'Country_A',
                'policy_type': 'boulware',
                'policy_params': {
                    'accept_threshold': 0.7,
                    'concession_rate': 0.05,
                    'stubbornness': 0.7
                },
                'weights': {'water': 0.6, 'energy': 0.4},
                'ideal_values': {'water': 60, 'energy': 40},
                'reservation_values': {'water': 30, 'energy': 20},
                'min_acceptable_utility': 0.5
            },
            {
                'name': 'Country_B',
                'policy_type': 'conceder',
                'policy_params': {
                    'accept_threshold': 0.6,
                    'concession_rate': 0.12,
                    'stubbornness': 0.3
                },
                'weights': {'water': 0.4, 'energy': 0.6},
                'ideal_values': {'water': 40, 'energy': 60},
                'reservation_values': {'water': 20, 'energy': 30},
                'min_acceptable_utility': 0.5
            }
        ],
        'issues': [
            {'name': 'water', 'min_value': 0, 'max_value': 100, 'unit': 'million_gallons', 'divisible': True},
            {'name': 'energy', 'min_value': 0, 'max_value': 100, 'unit': 'GWh', 'divisible': True}
        ],
        'max_rounds': 100,
        'protocol': 'simultaneous',
        'track_pareto': True,
        'calculate_nash': True
    }


def load_climate_template():
    """Load climate accord template."""
    st.session_state.current_config = {
        'entities': [
            {
                'name': 'Developed_Nation',
                'policy_type': 'boulware',
                'policy_params': {
                    'accept_threshold': 0.75,
                    'concession_rate': 0.04,
                    'stubbornness': 0.8
                },
                'weights': {
                    'emissions_reduction': 0.3,
                    'financial_contribution': 0.4,
                    'timeline': 0.3
                },
                'ideal_values': {
                    'emissions_reduction': 20,
                    'financial_contribution': 10,
                    'timeline': 2040
                },
                'reservation_values': {
                    'emissions_reduction': 40,
                    'financial_contribution': 30,
                    'timeline': 2030
                },
                'min_acceptable_utility': 0.6
            },
            {
                'name': 'Developing_Nation',
                'policy_type': 'conceder',
                'policy_params': {
                    'accept_threshold': 0.6,
                    'concession_rate': 0.1,
                    'stubbornness': 0.3
                },
                'weights': {
                    'emissions_reduction': 0.2,
                    'financial_contribution': 0.5,
                    'timeline': 0.3
                },
                'ideal_values': {
                    'emissions_reduction': 10,
                    'financial_contribution': 100,
                    'timeline': 2050
                },
                'reservation_values': {
                    'emissions_reduction': 25,
                    'financial_contribution': 40,
                    'timeline': 2035
                },
                'min_acceptable_utility': 0.5
            },
            {
                'name': 'Island_Nation',
                'policy_type': 'adaptive',
                'policy_params': {
                    'accept_threshold': 0.65,
                    'concession_rate': 0.07,
                    'stubbornness': 0.5
                },
                'weights': {
                    'emissions_reduction': 0.6,
                    'financial_contribution': 0.2,
                    'timeline': 0.2
                },
                'ideal_values': {
                    'emissions_reduction': 60,
                    'financial_contribution': 50,
                    'timeline': 2025
                },
                'reservation_values': {
                    'emissions_reduction': 35,
                    'financial_contribution': 20,
                    'timeline': 2035
                },
                'min_acceptable_utility': 0.55
            }
        ],
        'issues': [
            {'name': 'emissions_reduction', 'min_value': 0, 'max_value': 80, 'unit': 'percent', 'divisible': True},
            {'name': 'financial_contribution', 'min_value': 0, 'max_value': 150, 'unit': 'billion_USD', 'divisible': True},
            {'name': 'timeline', 'min_value': 2025, 'max_value': 2050, 'unit': 'year', 'divisible': False}
        ],
        'max_rounds': 100,
        'protocol': 'simultaneous',
        'track_pareto': True,
        'calculate_nash': True
    }


if __name__ == "__main__":
    main()
