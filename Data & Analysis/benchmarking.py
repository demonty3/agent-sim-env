"""
Performance benchmarking tools for the negotiation simulator.
Tests scalability, performance, and optimization opportunities.
"""

import time
import memory_profiler
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path

from models import (
    Entity, Issue, UtilityFunction, NegotiationPolicy,
    PolicyType, PolicyParameters, SimulationConfig
)
from protocol import NegotiationEngine, BatchNegotiationRunner
from utilities import analyze_negotiation_space
from advanced_strategies import create_advanced_strategy


# ===== BENCHMARK CONFIGURATIONS =====

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    name: str
    n_entities: List[int]
    n_issues: List[int]
    max_rounds: List[int]
    protocols: List[str]
    strategies: List[str]
    n_runs: int = 10
    parallel: bool = True


@dataclass
class BenchmarkResult:
    """Results from benchmark test."""
    config_name: str
    n_entities: int
    n_issues: int
    max_rounds: int
    protocol: str
    strategy: str
    execution_time: float
    memory_usage: float
    success_rate: float
    avg_rounds: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'config': self.config_name,
            'entities': self.n_entities,
            'issues': self.n_issues,
            'max_rounds': self.max_rounds,
            'protocol': self.protocol,
            'strategy': self.strategy,
            'time_seconds': self.execution_time,
            'memory_mb': self.memory_usage,
            'success_rate': self.success_rate,
            'avg_rounds': self.avg_rounds
        }


# ===== BENCHMARK SUITE =====

class NegotiationBenchmark:
    """Comprehensive benchmark suite for negotiation simulator."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./benchmarks")
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def run_scalability_test(self) -> pd.DataFrame:
        """Test scalability with increasing entities and issues."""

        config = BenchmarkConfig(
            name="Scalability Test",
            n_entities=[2, 3, 5, 8, 10],
            n_issues=[2, 5, 10, 15, 20],
            max_rounds=[50],
            protocols=["alternating", "simultaneous"],
            strategies=["linear_concession"],
            n_runs=5
        )

        results = self._run_benchmark_suite(config)
        df = pd.DataFrame([r.to_dict() for r in results])

        # Save results
        df.to_csv(self.output_dir / "scalability_results.csv", index=False)

        # Create visualizations
        self._plot_scalability_results(df)

        return df

    def run_strategy_comparison(self) -> pd.DataFrame:
        """Compare different negotiation strategies."""

        config = BenchmarkConfig(
            name="Strategy Comparison",
            n_entities=[2, 3],
            n_issues=[3],
            max_rounds=[100],
            protocols=["alternating"],
            strategies=[
                "linear_concession",
                "tit_for_tat",
                "boulware",
                "conceder",
                "adaptive",
                "mixed",
                "mcts"
            ],
            n_runs=20
        )

        results = self._run_benchmark_suite(config)
        df = pd.DataFrame([r.to_dict() for r in results])

        # Save results
        df.to_csv(self.output_dir / "strategy_comparison.csv", index=False)

        # Create visualizations
        self._plot_strategy_comparison(df)

        return df

    def run_protocol_benchmark(self) -> pd.DataFrame:
        """Compare different negotiation protocols."""

        config = BenchmarkConfig(
            name="Protocol Benchmark",
            n_entities=[2, 3, 5],
            n_issues=[5],
            max_rounds=[50, 100],
            protocols=["alternating", "simultaneous", "random"],
            strategies=["linear_concession"],
            n_runs=10
        )

        results = self._run_benchmark_suite(config)
        df = pd.DataFrame([r.to_dict() for r in results])

        # Save results
        df.to_csv(self.output_dir / "protocol_benchmark.csv", index=False)

        # Create visualizations
        self._plot_protocol_comparison(df)

        return df

    def run_memory_profile(self) -> Dict[str, Any]:
        """Profile memory usage for different configurations."""

        configs = [
            (2, 2, 50),   # Small
            (5, 5, 100),  # Medium
            (10, 10, 200) # Large
        ]

        memory_results = {}

        for n_entities, n_issues, max_rounds in configs:
            config_name = f"{n_entities}e_{n_issues}i_{max_rounds}r"

            # Create configuration
            entities = self._create_test_entities(n_entities)
            issues = self._create_test_issues(n_issues)
            config = SimulationConfig(
                entities=entities,
                issues=issues,
                max_rounds=max_rounds
            )

            # Profile memory
            @memory_profiler.profile
            def run_simulation():
                engine = NegotiationEngine(config)
                return engine.run()

            # Capture memory profile
            stream = io.StringIO()
            with memory_profiler.profile(stream=stream):
                outcome = run_simulation()

            memory_results[config_name] = {
                'peak_memory_mb': self._parse_memory_profile(stream.getvalue()),
                'success': outcome.success,
                'rounds': outcome.rounds_taken
            }

        # Save results
        import json
        with open(self.output_dir / "memory_profile.json", 'w') as f:
            json.dump(memory_results, f, indent=2)

        return memory_results

    def run_cpu_profile(self, n_entities: int = 3, n_issues: int = 5) -> str:
        """Profile CPU usage and identify bottlenecks."""

        # Create test configuration
        entities = self._create_test_entities(n_entities)
        issues = self._create_test_issues(n_issues)
        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=100
        )

        # Profile execution
        profiler = cProfile.Profile()
        profiler.enable()

        # Run multiple simulations
        for _ in range(10):
            engine = NegotiationEngine(config)
            engine.run()

        profiler.disable()

        # Generate report
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

        report = stream.getvalue()

        # Save report
        with open(self.output_dir / "cpu_profile.txt", 'w') as f:
            f.write(report)

        return report

    def run_parallel_benchmark(self) -> Dict[str, float]:
        """Benchmark parallel execution capabilities."""

        n_simulations = 100
        config = self._create_standard_config()

        # Sequential execution
        start = time.time()
        for _ in range(n_simulations):
            engine = NegotiationEngine(config)
            engine.run()
        sequential_time = time.time() - start

        # Thread-based parallel
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(n_simulations):
                future = executor.submit(self._run_single_simulation, config)
                futures.append(future)

            for future in futures:
                future.result()
        thread_time = time.time() - start

        # Process-based parallel
        start = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(n_simulations):
                future = executor.submit(self._run_single_simulation, config)
                futures.append(future)

            for future in futures:
                future.result()
        process_time = time.time() - start

        results = {
            'sequential_time': sequential_time,
            'thread_parallel_time': thread_time,
            'process_parallel_time': process_time,
            'thread_speedup': sequential_time / thread_time,
            'process_speedup': sequential_time / process_time
        }

        # Save results
        import json
        with open(self.output_dir / "parallel_benchmark.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_stress_test(self) -> Dict[str, Any]:
        """Run stress test with extreme configurations."""

        stress_configs = {
            'many_entities': (20, 3, 50),
            'many_issues': (3, 30, 50),
            'many_rounds': (3, 5, 1000),
            'complex': (10, 10, 500)
        }

        results = {}

        for test_name, (n_entities, n_issues, max_rounds) in stress_configs.items():
            try:
                entities = self._create_test_entities(n_entities)
                issues = self._create_test_issues(n_issues)
                config = SimulationConfig(
                    entities=entities,
                    issues=issues,
                    max_rounds=max_rounds,
                    track_pareto=False,  # Disable expensive calculations
                    calculate_nash=False
                )

                start = time.time()
                engine = NegotiationEngine(config)
                outcome = engine.run()
                execution_time = time.time() - start

                results[test_name] = {
                    'success': True,
                    'completed': outcome.success,
                    'rounds': outcome.rounds_taken,
                    'time': execution_time,
                    'error': None
                }

            except Exception as e:
                results[test_name] = {
                    'success': False,
                    'error': str(e)
                }

        # Save results
        import json
        with open(self.output_dir / "stress_test.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    # ===== HELPER METHODS =====

    def _run_benchmark_suite(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        results = []

        total_tests = (len(config.n_entities) * len(config.n_issues) *
                      len(config.max_rounds) * len(config.protocols) *
                      len(config.strategies))

        print(f"Running {total_tests} benchmark configurations...")

        for n_entities in config.n_entities:
            for n_issues in config.n_issues:
                for max_rounds in config.max_rounds:
                    for protocol in config.protocols:
                        for strategy in config.strategies:
                            result = self._run_single_benchmark(
                                config.name,
                                n_entities,
                                n_issues,
                                max_rounds,
                                protocol,
                                strategy,
                                config.n_runs
                            )
                            results.append(result)
                            print(f"  Completed: {n_entities}e_{n_issues}i_{protocol}_{strategy}")

        self.results.extend(results)
        return results

    def _run_single_benchmark(self,
                             config_name: str,
                             n_entities: int,
                             n_issues: int,
                             max_rounds: int,
                             protocol: str,
                             strategy: str,
                             n_runs: int) -> BenchmarkResult:
        """Run single benchmark configuration."""

        # Create test configuration
        entities = self._create_test_entities(n_entities, strategy)
        issues = self._create_test_issues(n_issues)

        config = SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=max_rounds,
            protocol=protocol,
            track_pareto=False,  # Disable for performance
            calculate_nash=False
        )

        # Run simulations
        start_time = time.time()
        start_memory = self._get_memory_usage()

        outcomes = []
        for _ in range(n_runs):
            engine = NegotiationEngine(config)
            outcome = engine.run()
            outcomes.append(outcome)

        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - start_memory

        # Calculate metrics
        success_rate = sum(1 for o in outcomes if o.success) / len(outcomes)
        avg_rounds = np.mean([o.rounds_taken for o in outcomes])

        return BenchmarkResult(
            config_name=config_name,
            n_entities=n_entities,
            n_issues=n_issues,
            max_rounds=max_rounds,
            protocol=protocol,
            strategy=strategy,
            execution_time=execution_time / n_runs,  # Per simulation
            memory_usage=memory_usage,
            success_rate=success_rate,
            avg_rounds=avg_rounds
        )

    def _create_test_entities(self,
                             n: int,
                             strategy: str = "linear_concession") -> List[Entity]:
        """Create test entities."""
        entities = []

        for i in range(n):
            # Create varied utility functions
            weights = {f"issue_{j}": np.random.random() for j in range(10)}
            ideal_values = {f"issue_{j}": np.random.uniform(20, 80) for j in range(10)}
            reservation_values = {f"issue_{j}": np.random.uniform(10, 90) for j in range(10)}

            # Create policy
            if strategy in ["adaptive", "mixed", "mcts", "q_learning"]:
                policy = create_advanced_strategy(strategy)
            else:
                policy = NegotiationPolicy(
                    type=PolicyType(strategy) if hasattr(PolicyType, strategy.upper())
                         else PolicyType.LINEAR_CONCESSION,
                    params=PolicyParameters(
                        accept_threshold=np.random.uniform(0.5, 0.8),
                        concession_rate=np.random.uniform(0.05, 0.15)
                    )
                )

            entity = Entity(
                name=f"Entity_{i}",
                utility_function=UtilityFunction(
                    weights=weights,
                    ideal_values=ideal_values,
                    reservation_values=reservation_values
                ),
                policy=policy
            )
            entities.append(entity)

        return entities

    def _create_test_issues(self, n: int) -> List[Issue]:
        """Create test issues."""
        return [
            Issue(
                name=f"issue_{i}",
                min_value=0,
                max_value=100,
                divisible=True
            )
            for i in range(n)
        ]

    def _create_standard_config(self) -> SimulationConfig:
        """Create standard test configuration."""
        entities = self._create_test_entities(3)
        issues = self._create_test_issues(5)

        return SimulationConfig(
            entities=entities,
            issues=issues,
            max_rounds=50
        )

    def _run_single_simulation(self, config: SimulationConfig):
        """Run single simulation (for parallel execution)."""
        engine = NegotiationEngine(config)
        return engine.run()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _parse_memory_profile(self, profile_output: str) -> float:
        """Parse memory profile output."""
        lines = profile_output.split('\n')
        max_memory = 0

        for line in lines:
            if 'MiB' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'MiB' in part and i > 0:
                        try:
                            memory = float(parts[i-1])
                            max_memory = max(max_memory, memory)
                        except:
                            pass

        return max_memory

    # ===== VISUALIZATION METHODS =====

    def _plot_scalability_results(self, df: pd.DataFrame):
        """Plot scalability test results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time vs entities
        pivot = df.pivot_table(values='time_seconds', index='entities',
                               columns='protocol', aggfunc='mean')
        pivot.plot(ax=axes[0, 0], marker='o')
        axes[0, 0].set_xlabel('Number of Entities')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].set_title('Scalability: Time vs Entities')

        # Time vs issues
        pivot = df.pivot_table(values='time_seconds', index='issues',
                               columns='protocol', aggfunc='mean')
        pivot.plot(ax=axes[0, 1], marker='o')
        axes[0, 1].set_xlabel('Number of Issues')
        axes[0, 1].set_ylabel('Execution Time (seconds)')
        axes[0, 1].set_title('Scalability: Time vs Issues')

        # Memory vs entities
        pivot = df.pivot_table(values='memory_mb', index='entities',
                               columns='protocol', aggfunc='mean')
        pivot.plot(ax=axes[1, 0], marker='o')
        axes[1, 0].set_xlabel('Number of Entities')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage vs Entities')

        # Success rate heatmap
        pivot = df.pivot_table(values='success_rate', index='entities',
                               columns='issues', aggfunc='mean')
        im = axes[1, 1].imshow(pivot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 1].set_xlabel('Number of Issues')
        axes[1, 1].set_ylabel('Number of Entities')
        axes[1, 1].set_title('Success Rate Heatmap')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / "scalability_plots.png", dpi=300)
        plt.close()

    def _plot_strategy_comparison(self, df: pd.DataFrame):
        """Plot strategy comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Success rate by strategy
        df.groupby('strategy')['success_rate'].mean().plot(
            kind='bar', ax=axes[0, 0]
        )
        axes[0, 0].set_xlabel('Strategy')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Success Rate by Strategy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Execution time by strategy
        df.groupby('strategy')['time_seconds'].mean().plot(
            kind='bar', ax=axes[0, 1]
        )
        axes[0, 1].set_xlabel('Strategy')
        axes[0, 1].set_ylabel('Execution Time (seconds)')
        axes[0, 1].set_title('Performance by Strategy')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Average rounds by strategy
        df.groupby('strategy')['avg_rounds'].mean().plot(
            kind='bar', ax=axes[1, 0]
        )
        axes[1, 0].set_xlabel('Strategy')
        axes[1, 0].set_ylabel('Average Rounds')
        axes[1, 0].set_title('Negotiation Length by Strategy')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Strategy comparison matrix
        strategies = df['strategy'].unique()
        matrix = np.zeros((len(strategies), 3))

        for i, strategy in enumerate(strategies):
            strategy_data = df[df['strategy'] == strategy]
            matrix[i, 0] = strategy_data['success_rate'].mean()
            matrix[i, 1] = 1 - (strategy_data['time_seconds'].mean() /
                                df['time_seconds'].max())  # Normalized
            matrix[i, 2] = 1 - (strategy_data['avg_rounds'].mean() /
                                df['avg_rounds'].max())  # Normalized

        im = axes[1, 1].imshow(matrix.T, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_xticks(range(len(strategies)))
        axes[1, 1].set_xticklabels(strategies, rotation=45)
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_yticklabels(['Success', 'Speed', 'Efficiency'])
        axes[1, 1].set_title('Strategy Performance Matrix')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(self.output_dir / "strategy_comparison.png", dpi=300)
        plt.close()

    def _plot_protocol_comparison(self, df: pd.DataFrame):
        """Plot protocol comparison results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Success rate by protocol and entities
        pivot = df.pivot_table(values='success_rate', index='entities',
                               columns='protocol', aggfunc='mean')
        pivot.plot(ax=axes[0], marker='o')
        axes[0].set_xlabel('Number of Entities')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Protocol Performance: Success Rate')

        # Time by protocol
        df.groupby('protocol')['time_seconds'].mean().plot(
            kind='bar', ax=axes[1]
        )
        axes[1].set_xlabel('Protocol')
        axes[1].set_ylabel('Execution Time (seconds)')
        axes[1].set_title('Protocol Performance: Speed')

        # Rounds by protocol
        df.groupby('protocol')['avg_rounds'].mean().plot(
            kind='bar', ax=axes[2]
        )
        axes[2].set_xlabel('Protocol')
        axes[2].set_ylabel('Average Rounds')
        axes[2].set_title('Protocol Performance: Efficiency')

        plt.tight_layout()
        plt.savefig(self.output_dir / "protocol_comparison.png", dpi=300)
        plt.close()

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("=" * 60)
        report.append("NEGOTIATION SIMULATOR BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")

        if self.results:
            df = pd.DataFrame([r.to_dict() for r in self.results])

            report.append("SUMMARY STATISTICS")
            report.append("-" * 40)
            report.append(f"Total benchmarks run: {len(self.results)}")
            report.append(f"Average execution time: {df['time_seconds'].mean():.3f} seconds")
            report.append(f"Average memory usage: {df['memory_mb'].mean():.2f} MB")
            report.append(f"Overall success rate: {df['success_rate'].mean():.2%}")
            report.append(f"Average rounds to completion: {df['avg_rounds'].mean():.1f}")
            report.append("")

            report.append("PERFORMANCE BY CONFIGURATION")
            report.append("-" * 40)

            # Group by entities
            entity_stats = df.groupby('entities').agg({
                'time_seconds': 'mean',
                'memory_mb': 'mean',
                'success_rate': 'mean'
            })
            report.append("\nBy Number of Entities:")
            report.append(entity_stats.to_string())

            # Group by protocol
            protocol_stats = df.groupby('protocol').agg({
                'time_seconds': 'mean',
                'success_rate': 'mean',
                'avg_rounds': 'mean'
            })
            report.append("\nBy Protocol:")
            report.append(protocol_stats.to_string())

            # Group by strategy
            if 'strategy' in df.columns:
                strategy_stats = df.groupby('strategy').agg({
                    'time_seconds': 'mean',
                    'success_rate': 'mean',
                    'avg_rounds': 'mean'
                })
                report.append("\nBy Strategy:")
                report.append(strategy_stats.to_string())

        report.append("")
        report.append("=" * 60)

        report_text = "\n".join(report)

        # Save report
        with open(self.output_dir / "benchmark_report.txt", 'w') as f:
            f.write(report_text)

        return report_text


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    print("Starting Negotiation Simulator Benchmarks...")

    benchmark = NegotiationBenchmark()

    # Run benchmarks
    print("\n1. Running scalability test...")
    scalability_df = benchmark.run_scalability_test()

    print("\n2. Running strategy comparison...")
    strategy_df = benchmark.run_strategy_comparison()

    print("\n3. Running protocol benchmark...")
    protocol_df = benchmark.run_protocol_benchmark()

    print("\n4. Running memory profile...")
    memory_results = benchmark.run_memory_profile()

    print("\n5. Running CPU profile...")
    cpu_report = benchmark.run_cpu_profile()

    print("\n6. Running parallel benchmark...")
    parallel_results = benchmark.run_parallel_benchmark()

    print("\n7. Running stress test...")
    stress_results = benchmark.run_stress_test()

    # Generate report
    print("\nGenerating comprehensive report...")
    report = benchmark.generate_report()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {benchmark.output_dir}")
    print("\nKey findings:")
    print(f"- Best protocol: {protocol_df.groupby('protocol')['success_rate'].mean().idxmax()}")
    print(f"- Fastest strategy: {strategy_df.groupby('strategy')['time_seconds'].mean().idxmin()}")
    print(f"- Process parallelization speedup: {parallel_results['process_speedup']:.2f}x")

    # Print stress test results
    print("\nStress test results:")
    for test_name, result in stress_results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result['success']:
            print(f"    Time: {result['time']:.2f}s, Rounds: {result['rounds']}")
