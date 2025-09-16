"""
Python SDK for the Negotiation Simulator API.
Provides a clean interface for programmatic access to all simulator features.
"""

import requests
import asyncio
import aiohttp
import websocket
import json
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from pathlib import Path


# ===== CONFIGURATION =====

class Config:
    """SDK Configuration."""
    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3
    WEBSOCKET_URL = "ws://localhost:8000/ws"


# ===== DATA CLASSES =====

@dataclass
class Issue:
    """Issue definition."""
    name: str
    min_value: float
    max_value: float
    divisible: bool = True
    unit: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class UtilityFunction:
    """Utility function definition."""
    weights: Dict[str, float]
    ideal_values: Dict[str, float]
    reservation_values: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Policy:
    """Negotiation policy."""
    type: str
    params: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Entity:
    """Negotiating entity."""
    name: str
    utility: UtilityFunction
    policy: Policy
    type: str = "company"
    min_acceptable_utility: float = 0.5

    def to_dict(self) -> dict:
        data = asdict(self)
        data['utility'] = self.utility.to_dict()
        data['policy'] = self.policy.to_dict()
        return data


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    entities: List[Entity]
    issues: List[Issue]
    max_rounds: int = 100
    protocol: str = "alternating"
    track_pareto: bool = True
    calculate_nash: bool = True
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'entities': [e.to_dict() for e in self.entities],
            'issues': [i.to_dict() for i in self.issues],
            'max_rounds': self.max_rounds,
            'protocol': self.protocol,
            'track_pareto': self.track_pareto,
            'calculate_nash': self.calculate_nash,
            'seed': self.seed
        }


@dataclass
class SimulationResult:
    """Simulation result."""
    id: str
    status: str
    success: Optional[bool] = None
    rounds_taken: Optional[int] = None
    final_agreement: Optional[Dict[str, float]] = None
    final_utilities: Optional[Dict[str, float]] = None
    impasse_reason: Optional[str] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationResult':
        """Create from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])

        # Handle nested outcome data
        if 'outcome' in data and data['outcome']:
            outcome = data['outcome']
            data.update({
                'success': outcome.get('success'),
                'rounds_taken': outcome.get('rounds_taken'),
                'final_agreement': outcome.get('final_agreement'),
                'final_utilities': outcome.get('final_utilities'),
                'impasse_reason': outcome.get('impasse_reason')
            })

        return cls(**{k: v for k, v in data.items()
                     if k in cls.__dataclass_fields__})


@dataclass
class BatchResult:
    """Batch simulation result."""
    batch_id: str
    n_runs: int
    success_rate: Optional[float] = None
    avg_rounds: Optional[float] = None
    avg_utility: Optional[float] = None
    status: str = "running"

    @classmethod
    def from_dict(cls, data: dict) -> 'BatchResult':
        """Create from dictionary."""
        if 'analysis' in data:
            analysis = data['analysis']
            data.update({
                'success_rate': analysis.get('success_rate'),
                'avg_rounds': analysis.get('average_rounds'),
                'avg_utility': analysis.get('average_utilities')
            })
        return cls(**{k: v for k, v in data.items()
                     if k in cls.__dataclass_fields__})


@dataclass
class AnalysisResult:
    """Negotiation space analysis result."""
    has_zopa: bool
    zopa_size: Optional[int]
    nash_solution: Optional[Dict[str, float]]
    nash_product: Optional[float]
    pareto_frontier_size: Optional[int]
    average_utilities: Optional[Dict[str, float]]

    @classmethod
    def from_dict(cls, data: dict) -> 'AnalysisResult':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items()
                     if k in cls.__dataclass_fields__})


# ===== EXCEPTIONS =====

class NegotiationAPIError(Exception):
    """Base exception for API errors."""
    pass


class SimulationTimeoutError(NegotiationAPIError):
    """Simulation timeout error."""
    pass


class InvalidConfigError(NegotiationAPIError):
    """Invalid configuration error."""
    pass


# ===== MAIN SDK CLASS =====

class NegotiationSimulatorSDK:
    """
    Python SDK for the Negotiation Simulator API.

    Example:
        sdk = NegotiationSimulatorSDK("http://api.negotiation-sim.ai")

        # Create configuration
        config = SimulationConfig(
            entities=[entity1, entity2],
            issues=[issue1, issue2]
        )

        # Run simulation
        result = sdk.simulate(config)
        print(f"Success: {result.success}")
    """

    def __init__(self,
                 base_url: str = Config.DEFAULT_BASE_URL,
                 timeout: int = Config.DEFAULT_TIMEOUT,
                 api_key: Optional[str] = None):
        """
        Initialize SDK.

        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key
        self.session = requests.Session()

        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'NegotiationSimulatorSDK/1.0'
        })

        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

    # ===== SIMULATION METHODS =====

    def simulate(self,
                config: Union[SimulationConfig, dict],
                wait: bool = True,
                poll_interval: float = 1.0) -> SimulationResult:
        """
        Run a single negotiation simulation.

        Args:
            config: Simulation configuration
            wait: Wait for completion if True
            poll_interval: Polling interval in seconds

        Returns:
            SimulationResult object
        """
        # Convert config to dict if needed
        if isinstance(config, SimulationConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config

        # Start simulation
        response = self._post('/simulate', json=config_dict)
        result = SimulationResult.from_dict(response)

        # Wait for completion if requested
        if wait and result.status == 'running':
            result = self._wait_for_completion(result.id, poll_interval)

        return result

    def simulate_batch(self,
                      config: Union[SimulationConfig, dict],
                      n_runs: int,
                      vary_params: Optional[Dict[str, Any]] = None,
                      wait: bool = True) -> BatchResult:
        """
        Run batch simulations.

        Args:
            config: Base configuration
            n_runs: Number of simulations
            vary_params: Parameters to vary
            wait: Wait for completion

        Returns:
            BatchResult object
        """
        if isinstance(config, SimulationConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config

        payload = {
            'config': config_dict,
            'n_runs': n_runs,
            'vary_params': vary_params
        }

        response = self._post('/batch', json=payload)
        batch_id = response['batch_id']

        if wait:
            return self._wait_for_batch(batch_id)

        return BatchResult(batch_id=batch_id, n_runs=n_runs)

    def get_simulation(self, simulation_id: str) -> SimulationResult:
        """Get simulation status and results."""
        response = self._get(f'/simulate/{simulation_id}')
        return SimulationResult.from_dict(response)

    def get_batch_results(self, batch_id: str) -> BatchResult:
        """Get batch simulation results."""
        response = self._get(f'/batch/{batch_id}')
        return BatchResult.from_dict(response)

    # ===== ANALYSIS METHODS =====

    def analyze_space(self,
                     entities: List[Entity],
                     issues: List[Issue],
                     samples: int = 1000) -> AnalysisResult:
        """
        Analyze negotiation space.

        Args:
            entities: List of entities
            issues: List of issues
            samples: Number of samples for analysis

        Returns:
            AnalysisResult object
        """
        payload = {
            'entities': [e.to_dict() for e in entities],
            'issues': [i.to_dict() for i in issues],
            'samples': samples
        }

        response = self._post('/analyze', json=payload)
        return AnalysisResult.from_dict(response)

    def get_advisor_recommendations(self, simulation_id: str) -> Dict[str, Any]:
        """Get AI advisor recommendations for a simulation."""
        return self._post(f'/advisor', json={'sim_id': simulation_id})

    # ===== CONFIGURATION MANAGEMENT =====

    def save_config(self, name: str, config: SimulationConfig) -> str:
        """
        Save configuration for reuse.

        Args:
            name: Configuration name
            config: Configuration to save

        Returns:
            Configuration ID
        """
        response = self._post('/configs',
                             params={'name': name},
                             json=config.to_dict())
        return response['config_id']

    def get_config(self, config_id: str) -> Dict[str, Any]:
        """Get saved configuration."""
        return self._get(f'/configs/{config_id}')

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all saved configurations."""
        return self._get('/configs')

    # ===== RESULTS MANAGEMENT =====

    def list_results(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List simulation results."""
        return self._get('/results', params={'limit': limit, 'offset': offset})

    def export_results(self,
                      simulation_id: str,
                      format: str = 'json',
                      output_path: Optional[Path] = None) -> Union[dict, bytes]:
        """
        Export simulation results.

        Args:
            simulation_id: Simulation ID
            format: Export format ('json' or 'csv')
            output_path: Optional path to save file

        Returns:
            Results data or bytes for CSV
        """
        response = self._get(f'/export/{simulation_id}',
                           params={'format': format},
                           raw=True)

        if output_path:
            with open(output_path, 'wb' if format == 'csv' else 'w') as f:
                if format == 'csv':
                    f.write(response.content)
                else:
                    json.dump(response.json(), f, indent=2)

        return response.content if format == 'csv' else response.json()

    # ===== HEALTH & METRICS =====

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._get('/health')

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return self._get('/metrics')

    # ===== WEBSOCKET METHODS =====

    def negotiate_realtime(self,
                          config: SimulationConfig,
                          on_round: Optional[Callable] = None,
                          on_complete: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run real-time negotiation via WebSocket.

        Args:
            config: Simulation configuration
            on_round: Callback for each round
            on_complete: Callback on completion

        Returns:
            Final negotiation result
        """
        ws_url = self.base_url.replace('http', 'ws') + '/ws'
        ws = websocket.WebSocket()
        ws.connect(ws_url)

        try:
            # Start negotiation
            ws.send(json.dumps({
                'type': 'start',
                'data': config.to_dict()
            }))

            # Process messages
            while True:
                message = json.loads(ws.recv())

                if message['type'] == 'round' and on_round:
                    on_round(message)
                elif message['type'] == 'completed':
                    if on_complete:
                        on_complete(message)
                    return message
                elif message['type'] == 'error':
                    raise NegotiationAPIError(message.get('message'))

        finally:
            ws.close()

    # ===== ASYNC METHODS =====

    async def simulate_async(self, config: SimulationConfig) -> SimulationResult:
        """Async version of simulate."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{self.base_url}/simulate',
                json=config.to_dict(),
                headers=self.session.headers
            ) as response:
                data = await response.json()
                return SimulationResult.from_dict(data)

    async def simulate_many_async(self,
                                 configs: List[SimulationConfig]) -> List[SimulationResult]:
        """Run multiple simulations concurrently."""
        tasks = [self.simulate_async(config) for config in configs]
        return await asyncio.gather(*tasks)

    # ===== PRIVATE METHODS =====

    def _get(self, endpoint: str, params: Optional[dict] = None, raw: bool = False):
        """Make GET request."""
        url = f'{self.base_url}{endpoint}'
        response = self.session.get(url, params=params, timeout=self.timeout)

        if response.status_code != 200:
            raise NegotiationAPIError(f"API error: {response.status_code} - {response.text}")

        return response if raw else response.json()

    def _post(self, endpoint: str, json: dict, params: Optional[dict] = None):
        """Make POST request."""
        url = f'{self.base_url}{endpoint}'
        response = self.session.post(url, json=json, params=params, timeout=self.timeout)

        if response.status_code not in [200, 201]:
            raise NegotiationAPIError(f"API error: {response.status_code} - {response.text}")

        return response.json()

    def _wait_for_completion(self,
                           simulation_id: str,
                           poll_interval: float = 1.0,
                           max_wait: float = 300) -> SimulationResult:
        """Wait for simulation to complete."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            result = self.get_simulation(simulation_id)

            if result.status in ['completed', 'failed']:
                return result

            time.sleep(poll_interval)

        raise SimulationTimeoutError(f"Simulation {simulation_id} timed out")

    def _wait_for_batch(self,
                       batch_id: str,
                       poll_interval: float = 5.0,
                       max_wait: float = 600) -> BatchResult:
        """Wait for batch to complete."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            result = self.get_batch_results(batch_id)

            if result.status != 'running':
                return result

            time.sleep(poll_interval)

        raise SimulationTimeoutError(f"Batch {batch_id} timed out")


# ===== CONVENIENCE FUNCTIONS =====

def create_simple_negotiation() -> SimulationConfig:
    """Create a simple two-party negotiation configuration."""

    # Define issues
    issues = [
        Issue("price", 100, 500, unit="USD"),
        Issue("quantity", 10, 100, unit="units")
    ]

    # Define entities
    buyer = Entity(
        name="Buyer",
        utility=UtilityFunction(
            weights={"price": 0.7, "quantity": 0.3},
            ideal_values={"price": 100, "quantity": 100},
            reservation_values={"price": 400, "quantity": 20}
        ),
        policy=Policy(
            type="linear_concession",
            params={"accept_threshold": 0.65, "concession_rate": 0.08}
        )
    )

    seller = Entity(
        name="Seller",
        utility=UtilityFunction(
            weights={"price": 0.8, "quantity": 0.2},
            ideal_values={"price": 500, "quantity": 10},
            reservation_values={"price": 200, "quantity": 80}
        ),
        policy=Policy(
            type="tit_for_tat",
            params={"accept_threshold": 0.6, "stubbornness": 0.5}
        )
    )

    return SimulationConfig(
        entities=[buyer, seller],
        issues=issues,
        max_rounds=50
    )


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    # Initialize SDK
    sdk = NegotiationSimulatorSDK()

    # Example 1: Simple negotiation
    print("Example 1: Simple Negotiation")
    print("-" * 40)

    config = create_simple_negotiation()
    result = sdk.simulate(config)

    print(f"Simulation ID: {result.id}")
    print(f"Success: {result.success}")
    print(f"Rounds: {result.rounds_taken}")
    print(f"Agreement: {result.final_agreement}")
    print(f"Utilities: {result.final_utilities}")
    print()

    # Example 2: Batch simulation
    print("Example 2: Batch Simulation")
    print("-" * 40)

    batch_result = sdk.simulate_batch(config, n_runs=10)
    print(f"Batch ID: {batch_result.batch_id}")
    print(f"Success Rate: {batch_result.success_rate:.1%}")
    print(f"Avg Rounds: {batch_result.avg_rounds:.1f}")
    print()

    # Example 3: Space analysis
    print("Example 3: Space Analysis")
    print("-" * 40)

    analysis = sdk.analyze_space(config.entities, config.issues)
    print(f"Has ZOPA: {analysis.has_zopa}")
    print(f"ZOPA Size: {analysis.zopa_size}")
    print(f"Nash Product: {analysis.nash_product:.3f}")
    print()

    # Example 4: Async simulations
    print("Example 4: Async Simulations")
    print("-" * 40)

    async def run_async_example():
        configs = [create_simple_negotiation() for _ in range(5)]
        results = await sdk.simulate_many_async(configs)

        success_count = sum(1 for r in results if r.success)
        print(f"Ran {len(results)} simulations asynchronously")
        print(f"Success rate: {success_count}/{len(results)}")

    asyncio.run(run_async_example())

    # Example 5: Real-time negotiation
    print("\nExample 5: Real-time Negotiation (WebSocket)")
    print("-" * 40)

    def on_round(message):
        print(f"Round {message['round']}: {message['offer']}")

    def on_complete(message):
        print(f"Negotiation complete: {message['success']}")

    # Uncomment to test WebSocket (requires running server)
    # result = sdk.negotiate_realtime(config, on_round, on_complete)

    print("\nSDK examples completed successfully!")
