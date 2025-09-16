"""
Background worker for processing long-running negotiation tasks.
Handles async processing via Redis queue.
"""

import asyncio
import json
import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import redis
from redis.exceptions import RedisError
import pickle

from config import settings
from models import SimulationConfig, Entity, Issue, UtilityFunction, NegotiationPolicy, PolicyParameters, PolicyType
from protocol import NegotiationEngine, BatchNegotiationRunner
from database import DatabaseOperations, SessionLocal

# Setup logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class NegotiationWorker:
    """Background worker for processing negotiation tasks."""

    def __init__(self):
        """Initialize worker with Redis connection."""
        self.redis_client = self._connect_redis()
        self.running = True
        self.current_task = None

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Worker initialized with {settings.WORKER_THREADS} threads")

    def _connect_redis(self) -> redis.Redis:
        """Establish Redis connection with retry logic."""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                client = redis.from_url(
                    settings.REDIS_URL,
                    **settings.get_redis_settings()
                )
                client.ping()
                logger.info("Connected to Redis successfully")
                return client
            except RedisError as e:
                logger.error(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

        if self.current_task:
            # Mark current task as interrupted
            self._update_task_status(
                self.current_task['id'],
                'interrupted',
                error="Worker shutdown"
            )

    async def process_tasks(self):
        """Main task processing loop."""
        logger.info("Starting task processing loop...")

        while self.running:
            try:
                # Get task from queue (blocking with timeout)
                task_data = self.redis_client.blpop(
                    ['negotiation_queue:high', 'negotiation_queue:normal', 'negotiation_queue:low'],
                    timeout=1
                )

                if task_data:
                    queue_name, task_json = task_data
                    task = json.loads(task_json)
                    self.current_task = task

                    logger.info(f"Processing task {task['id']} from {queue_name.decode()}")

                    # Process based on task type
                    if task['type'] == 'simulation':
                        await self._process_simulation(task)
                    elif task['type'] == 'batch':
                        await self._process_batch(task)
                    elif task['type'] == 'analysis':
                        await self._process_analysis(task)
                    else:
                        logger.error(f"Unknown task type: {task['type']}")
                        self._update_task_status(task['id'], 'failed', error='Unknown task type')

                    self.current_task = None

            except RedisError as e:
                logger.error(f"Redis error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                self.redis_client = self._connect_redis()

            except Exception as e:
                logger.error(f"Unexpected error in task processing: {e}")
                logger.error(traceback.format_exc())

                if self.current_task:
                    self._update_task_status(
                        self.current_task['id'],
                        'failed',
                        error=str(e)
                    )
                    self.current_task = None

                await asyncio.sleep(1)

        logger.info("Task processing loop ended")

    async def _process_simulation(self, task: Dict[str, Any]):
        """Process a single simulation task."""
        task_id = task['id']

        try:
            # Update status to running
            self._update_task_status(task_id, 'running')

            # Create configuration from task data
            config = SimulationConfig(**task['config'])

            # Run simulation with timeout
            start_time = time.time()

            # Run in executor to not block event loop
            loop = asyncio.get_event_loop()
            engine = NegotiationEngine(config)

            outcome = await asyncio.wait_for(
                loop.run_in_executor(None, engine.run),
                timeout=settings.SIMULATION_TIMEOUT
            )

            duration = time.time() - start_time

            # Store result
            result = {
                'task_id': task_id,
                'success': outcome.success,
                'rounds_taken': outcome.rounds_taken,
                'final_agreement': outcome.final_agreement,
                'final_utilities': outcome.final_utilities,
                'impasse_reason': outcome.impasse_reason,
                'duration': duration,
                'completed_at': datetime.now().isoformat()
            }

            # Save to Redis
            self.redis_client.setex(
                f"result:{task_id}",
                settings.REDIS_TTL,
                json.dumps(result)
            )

            # Save to database
            self._save_to_database(task_id, outcome, duration)

            # Update task status
            self._update_task_status(task_id, 'completed', result=result)

            logger.info(f"Task {task_id} completed successfully in {duration:.2f}s")

        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after {settings.SIMULATION_TIMEOUT}s")
            self._update_task_status(task_id, 'timeout')

        except Exception as e:
            logger.error(f"Error processing simulation {task_id}: {e}")
            logger.error(traceback.format_exc())
            self._update_task_status(task_id, 'failed', error=str(e))

    async def _process_batch(self, task: Dict[str, Any]):
        """Process a batch simulation task."""
        task_id = task['id']

        try:
            self._update_task_status(task_id, 'running')

            # Create base configuration
            config = SimulationConfig(**task['config'])
            n_runs = task.get('n_runs', 100)
            vary_params = task.get('vary_params')

            # Run batch with progress updates
            runner = BatchNegotiationRunner(config)
            results = []

            for i in range(n_runs):
                if not self.running:  # Check if worker is shutting down
                    break

                # Run single simulation
                loop = asyncio.get_event_loop()
                outcome = await loop.run_in_executor(
                    None,
                    runner.run_batch,
                    1,
                    vary_params
                )
                results.extend(outcome)

                # Update progress
                progress = (i + 1) / n_runs * 100
                self._update_task_progress(task_id, progress)

                # Yield to other tasks periodically
                if i % 10 == 0:
                    await asyncio.sleep(0)

            # Analyze results
            runner.results = results
            analysis = runner.analyze_results()

            # Store results
            result = {
                'task_id': task_id,
                'n_runs': len(results),
                'analysis': analysis,
                'completed_at': datetime.now().isoformat()
            }

            # Save to Redis
            self.redis_client.setex(
                f"batch_result:{task_id}",
                settings.REDIS_TTL,
                json.dumps(result)
            )

            # Update task status
            self._update_task_status(task_id, 'completed', result=result)

            logger.info(f"Batch task {task_id} completed: {len(results)} simulations")

        except Exception as e:
            logger.error(f"Error processing batch {task_id}: {e}")
            logger.error(traceback.format_exc())
            self._update_task_status(task_id, 'failed', error=str(e))

    async def _process_analysis(self, task: Dict[str, Any]):
        """Process a negotiation space analysis task."""
        task_id = task['id']

        try:
            self._update_task_status(task_id, 'running')

            # Import here to avoid circular dependency
            from utilities import analyze_negotiation_space

            # Build entities and issues from task payload (supports multiple shapes)
            entities = self._deserialize_entities(task['entities'])
            issues = self._deserialize_issues(task['issues'])
            samples = task.get('samples', 1000)

            # Run analysis
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None,
                analyze_negotiation_space,
                entities,
                issues,
                samples
            )

            # Store result
            result = {
                'task_id': task_id,
                'analysis': analysis,
                'completed_at': datetime.now().isoformat()
            }

            # Save to Redis
            self.redis_client.setex(
                f"analysis_result:{task_id}",
                settings.REDIS_TTL,
                json.dumps(result)
            )

            # Update task status
            self._update_task_status(task_id, 'completed', result=result)

            logger.info(f"Analysis task {task_id} completed")

        except Exception as e:
            logger.error(f"Error processing analysis {task_id}: {e}")
            logger.error(traceback.format_exc())
            self._update_task_status(task_id, 'failed', error=str(e))

    # ---------- Deserialization helpers ----------
    def _deserialize_entities(self, payload: Any) -> list[Entity]:
        entities: list[Entity] = []
        for e in payload:
            if isinstance(e, Entity):
                entities.append(e)
                continue

            # API-style shape: {'name','utility':{...}, 'policy':{'type','params'}}
            if 'utility' in e and 'policy' in e:
                util = e['utility']
                policy = e['policy']
                entity = Entity(
                    name=e['name'],
                    type=e.get('type', 'country'),
                    utility_function=UtilityFunction(
                        weights=util['weights'],
                        ideal_values=util['ideal_values'],
                        reservation_values=util['reservation_values']
                    ),
                    policy=NegotiationPolicy(
                        type=PolicyType(policy['type']),
                        params=PolicyParameters(**policy.get('params', {}))
                    ),
                    max_rounds=e.get('max_rounds', 100),
                    min_acceptable_utility=e.get('min_acceptable_utility', 0.5)
                )
                entities.append(entity)
                continue

            # Model-like dict shape
            if 'utility_function' in e and 'policy' in e:
                uf = e['utility_function']
                pol = e['policy']
                entity = Entity(
                    name=e['name'],
                    type=e.get('type', 'country'),
                    utility_function=UtilityFunction(**uf),
                    policy=NegotiationPolicy(
                        type=PolicyType(pol['type']) if isinstance(pol.get('type'), str) else pol['type'],
                        params=PolicyParameters(**pol.get('params', {}))
                    ),
                    max_rounds=e.get('max_rounds', 100),
                    min_acceptable_utility=e.get('min_acceptable_utility', 0.5)
                )
                entities.append(entity)
                continue

            raise ValueError(f"Unsupported entity payload: {e}")

        return entities

    def _deserialize_issues(self, payload: Any) -> list[Issue]:
        issues: list[Issue] = []
        for i in payload:
            if isinstance(i, Issue):
                issues.append(i)
                continue
            if {'name', 'min_value', 'max_value'} <= set(i.keys()):
                issues.append(
                    Issue(
                        name=i['name'],
                        min_value=i['min_value'],
                        max_value=i['max_value'],
                        divisible=i.get('divisible', True),
                        unit=i.get('unit')
                    )
                )
                continue
            raise ValueError(f"Unsupported issue payload: {i}")
        return issues

    def _update_task_status(self,
                           task_id: str,
                           status: str,
                           result: Optional[Dict] = None,
                           error: Optional[str] = None):
        """Update task status in Redis."""
        try:
            status_data = {
                'task_id': task_id,
                'status': status,
                'updated_at': datetime.now().isoformat()
            }

            if result:
                status_data['result'] = result
            if error:
                status_data['error'] = error

            # Update status
            self.redis_client.setex(
                f"task_status:{task_id}",
                settings.REDIS_TTL,
                json.dumps(status_data)
            )

            # Publish status update for real-time notifications
            self.redis_client.publish(
                f"task_updates:{task_id}",
                json.dumps(status_data)
            )

        except RedisError as e:
            logger.error(f"Failed to update task status: {e}")

    def _update_task_progress(self, task_id: str, progress: float):
        """Update task progress."""
        try:
            self.redis_client.setex(
                f"task_progress:{task_id}",
                settings.REDIS_TTL,
                json.dumps({
                    'task_id': task_id,
                    'progress': progress,
                    'updated_at': datetime.now().isoformat()
                })
            )
        except RedisError as e:
            logger.error(f"Failed to update task progress: {e}")

    def _save_to_database(self, task_id: str, outcome: Any, duration: float):
        """Save simulation results to database."""
        try:
            with SessionLocal() as session:
                db_ops = DatabaseOperations(session)

                simulation_data = {
                    'simulation_id': task_id,
                    'success': outcome.success,
                    'rounds_taken': outcome.rounds_taken,
                    'impasse_reason': outcome.impasse_reason,
                    'avg_utility': sum(outcome.final_utilities.values()) / len(outcome.final_utilities) if outcome.final_utilities else 0,
                    'final_agreement': outcome.final_agreement,
                    'final_utilities': outcome.final_utilities,
                    'pareto_optimal': outcome.pareto_optimal,
                    'nash_product': outcome.nash_bargaining_score,
                    'duration_seconds': duration
                }

                db_ops.save_simulation(simulation_data)
                logger.debug(f"Saved simulation {task_id} to database")

        except Exception as e:
            logger.error(f"Failed to save to database: {e}")


class WorkerPool:
    """Manage multiple worker processes."""

    def __init__(self, num_workers: int = None):
        """Initialize worker pool."""
        self.num_workers = num_workers or settings.WORKER_THREADS
        self.workers = []
        self.running = True

    async def start(self):
        """Start all workers."""
        logger.info(f"Starting worker pool with {self.num_workers} workers")

        for i in range(self.num_workers):
            worker = NegotiationWorker()
            task = asyncio.create_task(worker.process_tasks())
            self.workers.append((worker, task))
            logger.info(f"Started worker {i + 1}")

        # Wait for all workers
        try:
            await asyncio.gather(*[task for _, task in self.workers])
        except KeyboardInterrupt:
            logger.info("Worker pool interrupted")
            await self.shutdown()

    async def shutdown(self):
        """Shutdown all workers gracefully."""
        logger.info("Shutting down worker pool...")

        # Signal all workers to stop
        for worker, _ in self.workers:
            worker.running = False

        # Wait for tasks to complete
        await asyncio.gather(
            *[task for _, task in self.workers],
            return_exceptions=True
        )

        logger.info("Worker pool shutdown complete")


def main():
    """Main entry point for worker process."""
    logger.info("Starting Negotiation Worker Service")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Redis URL: {settings.REDIS_URL}")
    logger.info(f"Worker threads: {settings.WORKER_THREADS}")

    try:
        # Create and start worker pool
        pool = WorkerPool()
        asyncio.run(pool.start())

    except KeyboardInterrupt:
        logger.info("Worker service interrupted by user")
    except Exception as e:
        logger.error(f"Worker service failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Worker service stopped")


if __name__ == "__main__":
    main()
