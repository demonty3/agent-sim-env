"""
Database models and persistence layer using SQLAlchemy.
Stores simulation configurations, results, and analytics.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./negotiation_simulator.db")

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ===== DATABASE MODELS =====

class SimulationConfig(Base):
    """Store simulation configurations."""
    __tablename__ = "simulation_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    config_json = Column(JSON)  # Store full config as JSON

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String, nullable=True)

    # Relationships
    simulations = relationship("SimulationRun", back_populates="config")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'config': self.config_json,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class SimulationRun(Base):
    """Store individual simulation runs."""
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(String, unique=True, index=True)  # UUID
    config_id = Column(Integer, ForeignKey("simulation_configs.id"), nullable=True)

    # Simulation parameters
    max_rounds = Column(Integer)
    protocol = Column(String)

    # Outcome
    success = Column(Boolean)
    rounds_taken = Column(Integer)
    impasse_reason = Column(String, nullable=True)

    # Metrics
    avg_utility = Column(Float, nullable=True)
    min_utility = Column(Float, nullable=True)
    max_utility = Column(Float, nullable=True)
    pareto_optimal = Column(Boolean, nullable=True)
    nash_product = Column(Float, nullable=True)

    # Full data
    final_agreement = Column(JSON, nullable=True)
    final_utilities = Column(JSON, nullable=True)
    transcript_summary = Column(JSON, nullable=True)  # Compressed transcript

    # Metadata
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Relationships
    config = relationship("SimulationConfig", back_populates="simulations")
    entities = relationship("EntityResult", back_populates="simulation")
    rounds = relationship("NegotiationRoundRecord", back_populates="simulation")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'simulation_id': self.simulation_id,
            'config_id': self.config_id,
            'success': self.success,
            'rounds_taken': self.rounds_taken,
            'impasse_reason': self.impasse_reason,
            'avg_utility': self.avg_utility,
            'final_agreement': self.final_agreement,
            'final_utilities': self.final_utilities,
            'pareto_optimal': self.pareto_optimal,
            'nash_product': self.nash_product,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds
        }


class EntityResult(Base):
    """Store entity-specific results."""
    __tablename__ = "entity_results"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulation_runs.id"))

    # Entity info
    entity_name = Column(String)
    entity_type = Column(String)
    policy_type = Column(String)

    # Results
    final_utility = Column(Float)
    accept_threshold = Column(Float)
    concession_rate = Column(Float)
    total_concession = Column(Float, nullable=True)

    # Performance metrics
    offers_made = Column(Integer)
    offers_accepted = Column(Integer)
    offers_rejected = Column(Integer)

    # Relationships
    simulation = relationship("SimulationRun", back_populates="entities")


class NegotiationRoundRecord(Base):
    """Store round-by-round data."""
    __tablename__ = "negotiation_rounds"

    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulation_runs.id"))
    round_number = Column(Integer)

    # Round data
    proposer = Column(String)
    offer = Column(JSON)
    utilities = Column(JSON)  # Entity -> utility mapping
    responses = Column(JSON)  # Entity -> accept/reject
    status = Column(String)  # accepted/rejected/countered

    # Relationships
    simulation = relationship("SimulationRun", back_populates="rounds")


class BatchRun(Base):
    """Store batch simulation results."""
    __tablename__ = "batch_runs"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)

    # Batch parameters
    n_runs = Column(Integer)
    vary_params = Column(JSON, nullable=True)

    # Aggregate results
    success_rate = Column(Float)
    avg_rounds = Column(Float)
    avg_utility = Column(Float)
    std_utility = Column(Float)
    pareto_optimal_rate = Column(Float, nullable=True)

    # Timing
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    total_duration = Column(Float, nullable=True)

    # Full results
    results_summary = Column(JSON)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'name': self.name,
            'n_runs': self.n_runs,
            'success_rate': self.success_rate,
            'avg_rounds': self.avg_rounds,
            'avg_utility': self.avg_utility,
            'std_utility': self.std_utility,
            'results_summary': self.results_summary,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class AnalysisResult(Base):
    """Store negotiation space analysis results."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String, unique=True, index=True)

    # Analysis parameters
    n_entities = Column(Integer)
    n_issues = Column(Integer)
    samples = Column(Integer)

    # Results
    has_zopa = Column(Boolean)
    zopa_size = Column(Integer, nullable=True)
    pareto_frontier_size = Column(Integer, nullable=True)
    nash_solution = Column(JSON, nullable=True)
    nash_product = Column(Float, nullable=True)

    # Detailed analysis
    average_utilities = Column(JSON, nullable=True)
    max_joint_utility = Column(Float, nullable=True)
    min_joint_utility = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=func.now())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'n_entities': self.n_entities,
            'n_issues': self.n_issues,
            'has_zopa': self.has_zopa,
            'zopa_size': self.zopa_size,
            'nash_solution': self.nash_solution,
            'nash_product': self.nash_product,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AdvisoryReport(Base):
    """Store AI advisory reports."""
    __tablename__ = "advisory_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True)
    simulation_id = Column(Integer, ForeignKey("simulation_runs.id"), nullable=True)

    # Report content
    outcome_analysis = Column(Text)
    success_probability = Column(Float)
    parameter_suggestions = Column(JSON)
    strategy_suggestions = Column(JSON)
    key_insights = Column(JSON)

    # Metadata
    created_at = Column(DateTime, default=func.now())
    model_version = Column(String, nullable=True)


# ===== DATABASE OPERATIONS =====

class DatabaseOperations:
    """Database operations wrapper."""

    def __init__(self, session: Session):
        self.session = session

    def save_simulation(self, simulation_data: dict) -> SimulationRun:
        """Save simulation to database."""
        sim_run = SimulationRun(
            simulation_id=simulation_data.get('simulation_id'),
            success=simulation_data.get('success'),
            rounds_taken=simulation_data.get('rounds_taken'),
            impasse_reason=simulation_data.get('impasse_reason'),
            avg_utility=simulation_data.get('avg_utility'),
            final_agreement=simulation_data.get('final_agreement'),
            final_utilities=simulation_data.get('final_utilities'),
            pareto_optimal=simulation_data.get('pareto_optimal'),
            nash_product=simulation_data.get('nash_product'),
            completed_at=datetime.now()
        )

        self.session.add(sim_run)
        self.session.commit()
        self.session.refresh(sim_run)

        return sim_run

    def save_batch_results(self, batch_data: dict) -> BatchRun:
        """Save batch results to database."""
        batch_run = BatchRun(
            batch_id=batch_data.get('batch_id'),
            n_runs=batch_data.get('n_runs'),
            success_rate=batch_data.get('success_rate'),
            avg_rounds=batch_data.get('avg_rounds'),
            avg_utility=batch_data.get('avg_utility'),
            std_utility=batch_data.get('std_utility'),
            results_summary=batch_data.get('results_summary'),
            completed_at=datetime.now()
        )

        self.session.add(batch_run)
        self.session.commit()
        self.session.refresh(batch_run)

        return batch_run

    def get_simulations(self,
                       limit: int = 100,
                       offset: int = 0,
                       success_only: bool = False) -> List[SimulationRun]:
        """Get simulations from database."""
        query = self.session.query(SimulationRun)

        if success_only:
            query = query.filter(SimulationRun.success == True)

        return query.order_by(SimulationRun.created_at.desc()).offset(offset).limit(limit).all()

    def get_simulation_by_id(self, simulation_id: str) -> Optional[SimulationRun]:
        """Get simulation by ID."""
        return self.session.query(SimulationRun).filter(
            SimulationRun.simulation_id == simulation_id
        ).first()

    def get_batch_results(self, batch_id: str) -> Optional[BatchRun]:
        """Get batch results by ID."""
        return self.session.query(BatchRun).filter(
            BatchRun.batch_id == batch_id
        ).first()

    def get_statistics(self) -> dict:
        """Get database statistics."""
        total_simulations = self.session.query(SimulationRun).count()
        successful_simulations = self.session.query(SimulationRun).filter(
            SimulationRun.success == True
        ).count()

        avg_rounds = self.session.query(func.avg(SimulationRun.rounds_taken)).scalar()
        avg_utility = self.session.query(func.avg(SimulationRun.avg_utility)).scalar()

        return {
            'total_simulations': total_simulations,
            'successful_simulations': successful_simulations,
            'success_rate': successful_simulations / total_simulations if total_simulations > 0 else 0,
            'avg_rounds': avg_rounds or 0,
            'avg_utility': avg_utility or 0
        }

    def cleanup_old_records(self, days: int = 30):
        """Clean up old records."""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)

        # Delete old simulation runs
        deleted = self.session.query(SimulationRun).filter(
            SimulationRun.created_at < cutoff_date
        ).delete()

        self.session.commit()
        return deleted


# ===== UTILITY FUNCTIONS =====

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def migrate_database():
    """Run database migrations."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    print("Database migrated successfully")


# ===== CACHING LAYER =====

class CacheManager:
    """Redis-based caching for performance."""

    def __init__(self, redis_url: Optional[str] = None):
        import redis
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = redis.from_url(self.redis_url)
        self.ttl = 3600  # 1 hour default TTL

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        self.client.setex(
            key,
            ttl or self.ttl,
            json.dumps(value)
        )

    def delete(self, key: str):
        """Delete key from cache."""
        self.client.delete(key)

    def clear(self):
        """Clear all cache."""
        self.client.flushdb()


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Initialize database
    init_database()

    # Example: Save a simulation
    with SessionLocal() as session:
        db_ops = DatabaseOperations(session)

        # Save simulation
        sim_data = {
            'simulation_id': 'test-123',
            'success': True,
            'rounds_taken': 25,
            'avg_utility': 0.75,
            'final_agreement': {'price': 150, 'quantity': 500},
            'final_utilities': {'Buyer': 0.7, 'Seller': 0.8}
        }

        sim_run = db_ops.save_simulation(sim_data)
        print(f"Saved simulation: {sim_run.id}")

        # Get statistics
        stats = db_ops.get_statistics()
        print(f"Database statistics: {stats}")
