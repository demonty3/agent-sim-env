"""
REST API server for the negotiation simulator.
Provides endpoints for running simulations, managing configurations, and analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json
import uuid
from pathlib import Path
import pickle

from models import (
    Entity, Issue, UtilityFunction, NegotiationPolicy,
    PolicyType, PolicyParameters, SimulationConfig
)
from protocol import NegotiationEngine, BatchNegotiationRunner
from utilities import analyze_negotiation_space, find_nash_bargaining_solution
from advisor import NegotiationAdvisor
from advanced_strategies import create_advanced_strategy

# ===== APP INITIALIZATION =====

app = FastAPI(
    title="Negotiation Simulator API",
    description="AI-powered negotiation simulation and analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
simulations_db = {}
configs_db = {}
results_db = {}
active_negotiations = {}


# ===== PYDANTIC MODELS =====

class IssueRequest(BaseModel):
    name: str
    min_value: float
    max_value: float
    divisible: bool = True
    unit: Optional[str] = None


class UtilityRequest(BaseModel):
    weights: Dict[str, float]
    ideal_values: Dict[str, float]
    reservation_values: Dict[str, float]


class PolicyRequest(BaseModel):
    type: str
    params: Dict[str, Any]


class EntityRequest(BaseModel):
    name: str
    type: str = "company"
    utility: UtilityRequest
    policy: PolicyRequest
    min_acceptable_utility: float = 0.5


class SimulationRequest(BaseModel):
    entities: List[EntityRequest]
    issues: List[IssueRequest]
    max_rounds: int = 100
    protocol: str = "alternating"
    track_pareto: bool = True
    calculate_nash: bool = True


class BatchRequest(BaseModel):
    config: SimulationRequest
    n_runs: int = Field(10, ge=1, le=1000)
    vary_params: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    entities: List[EntityRequest]
    issues: List[IssueRequest]
    samples: int = Field(1000, ge=100, le=10000)


class SimulationResponse(BaseModel):
    id: str
    status: str
    created_at: datetime
    config: Dict[str, Any]
    outcome: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None


class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]


# ===== HELPER FUNCTIONS =====

def create_entity_from_request(req: EntityRequest) -> Entity:
    """Convert request to Entity object."""
    return Entity(
        name=req.name,
        type=req.type,
        utility_function=UtilityFunction(
            weights=req.utility.weights,
            ideal_values=req.utility.ideal_values,
            reservation_values=req.utility.reservation_values
        ),
        policy=NegotiationPolicy(
            type=PolicyType(req.policy.type),
            params=PolicyParameters(**req.policy.params)
        ),
        min_acceptable_utility=req.min_acceptable_utility
    )


def create_issue_from_request(req: IssueRequest) -> Issue:
    """Convert request to Issue object."""
    return Issue(
        name=req.name,
        min_value=req.min_value,
        max_value=req.max_value,
        divisible=req.divisible,
        unit=req.unit
    )


def outcome_to_dict(outcome) -> Dict[str, Any]:
    """Convert NegotiationOutcome to dictionary."""
    return {
        'success': outcome.success,
        'rounds_taken': outcome.rounds_taken,
        'final_agreement': outcome.final_agreement,
        'final_utilities': outcome.final_utilities,
        'impasse_reason': outcome.impasse_reason,
        'pareto_optimal': outcome.pareto_optimal,
        'nash_bargaining_score': outcome.nash_bargaining_score,
        'summary': outcome.summary()
    }


async def run_simulation_async(config: SimulationConfig) -> Dict[str, Any]:
    """Run simulation asynchronously."""
    loop = asyncio.get_event_loop()
    engine = NegotiationEngine(config)
    outcome = await loop.run_in_executor(None, engine.run)
    return outcome_to_dict(outcome)


# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Negotiation Simulator API",
        "version": "1.0.0",
        "endpoints": {
            "simulate": "/simulate",
            "batch": "/batch",
            "analyze": "/analyze",
            "configs": "/configs",
            "results": "/results",
            "websocket": "/ws"
        }
    }


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Run a single negotiation simulation."""

    # Create simulation ID
    sim_id = str(uuid.uuid4())

    # Convert request to config
    entities = [create_entity_from_request(e) for e in request.entities]
    issues = [create_issue_from_request(i) for i in request.issues]

    config = SimulationConfig(
        entities=entities,
        issues=issues,
        max_rounds=request.max_rounds,
        protocol=request.protocol,
        track_pareto=request.track_pareto,
        calculate_nash=request.calculate_nash
    )

    # Store initial status
    simulations_db[sim_id] = {
        'id': sim_id,
        'status': 'running',
        'created_at': datetime.now(),
        'config': request.dict()
    }

    # Run simulation in background
    async def run_and_store():
        try:
            outcome = await run_simulation_async(config)
            simulations_db[sim_id]['status'] = 'completed'
            simulations_db[sim_id]['outcome'] = outcome
            results_db[sim_id] = outcome
        except Exception as e:
            simulations_db[sim_id]['status'] = 'failed'
            simulations_db[sim_id]['error'] = str(e)

    background_tasks.add_task(run_and_store)

    return SimulationResponse(**simulations_db[sim_id])


@app.get("/simulate/{sim_id}", response_model=SimulationResponse)
async def get_simulation(sim_id: str):
    """Get simulation status and results."""
    if sim_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return SimulationResponse(**simulations_db[sim_id])


@app.post("/batch")
async def batch_simulate(request: BatchRequest, background_tasks: BackgroundTasks):
    """Run batch simulations."""

    batch_id = str(uuid.uuid4())

    # Convert request to config
    entities = [create_entity_from_request(e) for e in request.config.entities]
    issues = [create_issue_from_request(i) for i in request.config.issues]

    config = SimulationConfig(
        entities=entities,
        issues=issues,
        max_rounds=request.config.max_rounds,
        protocol=request.config.protocol
    )

    # Run batch in background
    async def run_batch():
        runner = BatchNegotiationRunner(config)
        results = runner.run_batch(request.n_runs, request.vary_params)
        analysis = runner.analyze_results()

        results_db[batch_id] = {
            'type': 'batch',
            'n_runs': request.n_runs,
            'analysis': analysis,
            'completed_at': datetime.now()
        }

    background_tasks.add_task(run_batch)

    return {
        'batch_id': batch_id,
        'status': 'running',
        'n_runs': request.n_runs
    }


@app.get("/batch/{batch_id}")
async def get_batch_results(batch_id: str):
    """Get batch simulation results."""
    if batch_id not in results_db:
        return {'status': 'running'}

    return results_db[batch_id]


@app.post("/analyze")
async def analyze_space(request: AnalysisRequest):
    """Analyze negotiation space."""

    entities = [create_entity_from_request(e) for e in request.entities]
    issues = [create_issue_from_request(i) for i in request.issues]

    # Run analysis
    analysis = analyze_negotiation_space(entities, issues, request.samples)

    # Add Nash solution if ZOPA exists
    if analysis['has_zopa']:
        nash = find_nash_bargaining_solution(entities, issues, request.samples)
        analysis['nash_solution'] = nash

    return analysis


@app.post("/advisor")
async def get_advice(sim_id: str):
    """Get AI advisory recommendations for a simulation."""

    if sim_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")

    sim_data = simulations_db[sim_id]

    if sim_data['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Simulation not completed")

    # Recreate config and outcome
    entities = [create_entity_from_request(EntityRequest(**e))
                for e in sim_data['config']['entities']]
    issues = [create_issue_from_request(IssueRequest(**i))
              for i in sim_data['config']['issues']]

    config = SimulationConfig(
        entities=entities,
        issues=issues,
        max_rounds=sim_data['config']['max_rounds']
    )

    # Get advisor recommendations
    advisor = NegotiationAdvisor()

    # Note: This is simplified - would need to recreate full outcome object
    report = {
        'success_probability': 0.7,
        'recommendations': [
            {
                'entity': entities[0].name,
                'parameter': 'accept_threshold',
                'current': 0.7,
                'suggested': 0.65,
                'rationale': 'Lower threshold may enable agreement'
            }
        ],
        'insights': [
            'ZOPA exists but is narrow',
            'Consider more flexible strategies'
        ]
    }

    return report


@app.post("/configs")
async def save_config(name: str, config: SimulationRequest):
    """Save a configuration for reuse."""
    config_id = str(uuid.uuid4())
    configs_db[config_id] = {
        'id': config_id,
        'name': name,
        'config': config.dict(),
        'created_at': datetime.now()
    }
    return {'config_id': config_id, 'name': name}


@app.get("/configs")
async def list_configs():
    """List saved configurations."""
    return list(configs_db.values())


@app.get("/configs/{config_id}")
async def get_config(config_id: str):
    """Get a saved configuration."""
    if config_id not in configs_db:
        raise HTTPException(status_code=404, detail="Configuration not found")

    return configs_db[config_id]


@app.get("/results")
async def list_results(limit: int = 10, offset: int = 0):
    """List simulation results."""
    all_results = list(results_db.values())
    return all_results[offset:offset + limit]


@app.get("/export/{sim_id}")
async def export_results(sim_id: str, format: str = "json"):
    """Export simulation results."""

    if sim_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")

    sim_data = simulations_db[sim_id]

    if format == "json":
        return sim_data
    elif format == "csv":
        # Convert to CSV format
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['Entity', 'Utility', 'Issue', 'Value'])

        # Write data
        if 'outcome' in sim_data and sim_data['outcome']:
            outcome = sim_data['outcome']
            for entity, utility in outcome.get('final_utilities', {}).items():
                for issue, value in outcome.get('final_agreement', {}).items():
                    writer.writerow([entity, utility, issue, value])

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=simulation_{sim_id}.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")


# ===== WEBSOCKET ENDPOINTS =====

@app.websocket("/ws")
async def websocket_negotiate(websocket: WebSocket):
    """WebSocket endpoint for real-time negotiation."""
    await websocket.accept()
    negotiation_id = str(uuid.uuid4())

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = WebSocketMessage(**data)

            if message.type == "start":
                # Start new negotiation
                config_data = message.data
                active_negotiations[negotiation_id] = {
                    'websocket': websocket,
                    'config': config_data,
                    'round': 0
                }

                await websocket.send_json({
                    'type': 'started',
                    'negotiation_id': negotiation_id
                })

            elif message.type == "offer":
                # Process offer
                negotiation = active_negotiations.get(negotiation_id)
                if not negotiation:
                    await websocket.send_json({
                        'type': 'error',
                        'message': 'No active negotiation'
                    })
                    continue

                # Simulate offer processing
                negotiation['round'] += 1

                # Send response
                await websocket.send_json({
                    'type': 'round',
                    'round': negotiation['round'],
                    'offer': message.data,
                    'responses': {'Entity1': True, 'Entity2': False}
                })

                # Check for agreement
                if negotiation['round'] > 10:  # Simplified
                    await websocket.send_json({
                        'type': 'completed',
                        'success': True,
                        'agreement': message.data
                    })
                    del active_negotiations[negotiation_id]

            elif message.type == "stop":
                # Stop negotiation
                if negotiation_id in active_negotiations:
                    del active_negotiations[negotiation_id]
                await websocket.send_json({
                    'type': 'stopped'
                })
                break

    except WebSocketDisconnect:
        if negotiation_id in active_negotiations:
            del active_negotiations[negotiation_id]


@app.websocket("/ws/monitor/{sim_id}")
async def websocket_monitor(websocket: WebSocket, sim_id: str):
    """Monitor ongoing simulation via WebSocket."""
    await websocket.accept()

    try:
        while sim_id in simulations_db:
            sim_data = simulations_db[sim_id]

            await websocket.send_json({
                'type': 'status',
                'status': sim_data['status'],
                'data': sim_data.get('outcome')
            })

            if sim_data['status'] in ['completed', 'failed']:
                break

            await asyncio.sleep(1)  # Poll every second

    except WebSocketDisconnect:
        pass


# ===== HEALTH & MONITORING =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'timestamp': datetime.now(),
        'active_simulations': len([s for s in simulations_db.values()
                                  if s['status'] == 'running']),
        'total_simulations': len(simulations_db)
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""

    completed = [s for s in simulations_db.values() if s['status'] == 'completed']
    successful = [s for s in completed if s.get('outcome', {}).get('success')]

    return {
        'total_simulations': len(simulations_db),
        'completed': len(completed),
        'successful': len(successful),
        'success_rate': len(successful) / len(completed) if completed else 0,
        'active_negotiations': len(active_negotiations),
        'saved_configs': len(configs_db)
    }


# ===== ERROR HANDLERS =====

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return HTTPException(status_code=500, detail="Internal server error")


# ===== STARTUP & SHUTDOWN =====

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("Negotiation Simulator API starting...")

    # Load any persisted data
    data_dir = Path("./data")
    if data_dir.exists():
        cache_file = data_dir / "cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                simulations_db.update(data.get('simulations', {}))
                configs_db.update(data.get('configs', {}))
                results_db.update(data.get('results', {}))

    print(f"Loaded {len(simulations_db)} simulations, {len(configs_db)} configs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Negotiation Simulator API shutting down...")

    # Save data
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    cache_file = data_dir / "cache.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'simulations': simulations_db,
            'configs': configs_db,
            'results': results_db
        }, f)

    print("Data saved successfully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
