# AI Agent Negotiation Simulator

A sophisticated multi-agent negotiation simulator for modeling complex bargaining scenarios between countries, companies, or other entities.

## Features

- **Multiple negotiation protocols**: Alternating offers, simultaneous proposals, random selection
- **Game theory analysis**: Nash bargaining, Pareto optimality, ZOPA detection
- **Flexible policies**: Linear concession, tit-for-tat, Boulware (hard), Conceder (soft), Adaptive
- **Rich visualizations**: Track utility progression, analyze outcomes
- **Batch simulations**: Run parameter sweeps and statistical analysis
- **LLM Integration Ready**: Framework for AI-assisted parameter tuning (V2)

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pydantic numpy pandas matplotlib typer pyyaml rich jupyter pytest
```

### Run the Demo

```bash
# Interactive demo
python main.py

# Quick test
python main.py quick

# CLI mode
python main.py cli
```

### Use the CLI

```bash
# Run a single negotiation
python cli.py run sample_config.yaml --viz

# Run batch simulations
python cli.py batch sample_config.yaml --n-runs 100

# Analyze negotiation space
python cli.py analyze sample_config.yaml --samples 1000

# Generate example configs
python cli.py example
```

## Project Structure

```
negotiation-simulator/
├── models.py           # Core data models (Entity, Policy, Utility)
├── utilities.py        # Game theory calculations
├── protocol.py         # Negotiation engine
├── cli.py             # Command-line interface
├── main.py            # Demo runner
├── sample_config.yaml # Example configuration
└── examples/          # Generated example configs
```

## Core Concepts

### Entities
Negotiating parties (countries, companies) with:
- **Utility functions**: How they value different outcomes
- **Policies**: Their negotiation strategy
- **Constraints**: Minimum acceptable outcomes

### Issues
The topics being negotiated (price, quantity, resources) with:
- **Range**: Min/max values
- **Divisibility**: Continuous or discrete
- **Units**: For clarity

### Policies
How entities negotiate:
- **Linear Concession**: Gradual compromise over time
- **Tit-for-Tat**: Mirror opponent's concessions
- **Boulware**: Hard early, concede late
- **Conceder**: Concede early, hard late
- **Adaptive**: Learn from opponent behavior

### Protocols
How negotiations proceed:
- **Alternating**: Take turns making offers
- **Simultaneous**: All propose, best selected
- **Random**: Random proposer each round

## Configuration Format

Create YAML files to define scenarios:

```yaml
entities:
  - name: "Buyer"
    utility:
      weights: {price: 0.7, quantity: 0.3}
      ideal_values: {price: 100, quantity: 1000}
      reservation_values: {price: 150, quantity: 500}
    policy:
      type: "linear_concession"
      params:
        accept_threshold: 0.7
        concession_rate: 0.1

issues:
  - name: "price"
    min_value: 50
    max_value: 200
    unit: "USD"

max_rounds: 50
protocol: "alternating"
```

## Example Usage

### Simple Two-Party Negotiation

```python
from models import Entity, Issue, UtilityFunction, NegotiationPolicy
from protocol import quick_negotiate

# Define issues
issues = [
    Issue(name="price", min_value=100, max_value=500)
]

# Create entities
buyer = Entity(
    name="Buyer",
    utility_function=UtilityFunction(
        weights={"price": 1.0},
        ideal_values={"price": 100},
        reservation_values={"price": 400}
    ),
    policy=NegotiationPolicy(type="linear_concession")
)

seller = Entity(
    name="Seller",
    utility_function=UtilityFunction(
        weights={"price": 1.0},
        ideal_values={"price": 500},
        reservation_values={"price": 200}
    ),
    policy=NegotiationPolicy(type="tit_for_tat")
)

# Run negotiation
outcome = quick_negotiate([buyer, seller], issues)
print(outcome.summary())
```

### Batch Analysis

```python
from protocol import BatchNegotiationRunner
from models import SimulationConfig

config = SimulationConfig(...)  # Your config
runner = BatchNegotiationRunner(config)

# Run 100 simulations
results = runner.run_batch(100)
analysis = runner.analyze_results()

print(f"Success rate: {analysis['success_rate']:.1%}")
```

## Advanced Features

### Game Theory Analysis

The simulator calculates:
- **ZOPA**: Zone of Possible Agreement
- **Nash Bargaining Solution**: Theoretically optimal outcome
- **Pareto Frontier**: Non-dominated solutions
- **Bargaining Power**: Based on BATNA and patience

### Parameter Tuning (V2)

Future LLM integration will enable:
- Automatic policy parameter optimization
- Strategy recommendations based on opponent behavior
- Natural language explanations of outcomes

## Development Roadmap

### Current (V1)
✅ Core negotiation engine
✅ Multiple policies and protocols
✅ Game theory analysis
✅ CLI and visualization
✅ Batch simulations

### Planned (V2)
- [ ] LLM advisory layer for parameter tuning
- [ ] Coalition formation
- [ ] Incomplete information scenarios
- [ ] Side payments
- [ ] Multi-round package deals
- [ ] Real-time strategy adaptation

## Tips for Good Simulations

1. **Ensure ZOPA exists**: Check that reservation values allow for agreement
2. **Balance patience**: Too stubborn = impasse, too eager = poor deals
3. **Vary concession rates**: Test different speeds of compromise
4. **Use appropriate protocols**: Alternating for bilateral, simultaneous for multilateral
5. **Analyze the space first**: Use the analyze command before running

## Troubleshooting

**No ZOPA Found**: Reservation values are incompatible. Adjust ideal/reservation values.

**Constant Impasses**: Reduce accept_threshold or increase concession_rate.

**Too Quick Agreement**: Increase accept_threshold or reduce initial concession.

**Deadlock**: Entities are repeating same offers. Add randomness or adjust stubbornness.

## Contributing

Feel free to extend the simulator with:
- New negotiation policies
- Additional utility function types
- More sophisticated protocols
- Domain-specific scenarios

## License

MIT License - Use freely for research and education.

---

Built with 🤝 for modeling complex negotiations in international relations, business, and resource allocation.
