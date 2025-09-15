Title: Stabilize negotiation engine, fix Issue validation, and improve ZOPA analysis

Summary:
- Enforces valid issue ranges, ensures proposers auto-accept their own offers across protocols, and normalizes ZOPA size to stabilize advisor probabilities. Adjusts default `min_acceptable_utility` to 0.1 for robust ZOPA sampling in unspecified scenarios. Adds Pareto optimal rate to batch analysis.

Motivation:
- Tests revealed missing cross-field validation and protocol behavior preventing agreements.
- Advisor success probability depended on raw sampling counts and was too conservative.
- CLI batch expected a Pareto rate metric not present in analysis results.

Changes:
- models.py
  - Add Pydantic v2 `model_validator` (with v1 fallback) to enforce `Issue.max_value > min_value`.
  - Set default `Entity.min_acceptable_utility` to 0.1 (YAML configs can override; improves ZOPA sampling when unspecified).
- protocol.py
  - Proposers auto-accept their own offers in alternating, simultaneous, and random protocols.
  - BatchNegotiationRunner.analyze_results: add `pareto_optimal_rate` and include it in returned metrics.
- utilities.py
  - Normalize `zopa_size` to a 0–100 scale relative to `samples`.

Validation:
- Tests: `24 passed, 1 warning` via `python -m pytest -q` in `negotiation-simulator/`.
- Demo: `python negotiation-simulator/main.py quick` — agreement reached in 4 rounds.
- CLI:
  - `run` on sample YAML executed end-to-end (impasse for a stringent multi-party scenario).
  - `batch` on sample YAML now returns a formatted table including Pareto Optimal Rate.

Backward Compatibility:
- YAML-configured `min_acceptable_utility` values remain authoritative; only the unspecified default changed.
- Auto-accept by proposers aligns with typical negotiation flows and increases likelihood of agreements without breaking API.

Risks:
- Lower default `min_acceptable_utility` makes default-constructed scenarios more permissive; mitigated by explicit configuration and improved advisor interpretability.

Follow-ups:
- Document guidance for setting `accept_threshold` vs `min_acceptable_utility`.
- Optionally set `MPLCONFIGDIR` in CLI to silence Matplotlib cache warnings in restricted environments.
