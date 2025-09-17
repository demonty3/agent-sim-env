# Makefile for AI Agent Negotiation Simulator
# Provides convenient commands for development, testing, and deployment

.PHONY: help install test clean run docker docs deploy benchmark

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := negotiation-simulator
DOCKER_IMAGE := $(PROJECT_NAME):latest
PORT := 8501

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(GREEN)AI Agent Negotiation Simulator - Available Commands$(NC)"
	@echo "=================================================="
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make install      - Install all dependencies"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make clean        - Clean cache and temporary files"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run linting checks"
	@echo "  make type-check   - Run type checking with mypy"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make test-fast    - Run fast tests only"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo ""
	@echo "$(YELLOW)Running:$(NC)"
	@echo "  make run          - Run interactive demo"
	@echo "  make cli          - Run CLI interface"
	@echo "  make web          - Run web interface"
	@echo "  make api          - Run API server"
	@echo "  make jupyter      - Start Jupyter notebook"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-up    - Start all services with docker-compose"
	@echo "  make docker-down  - Stop all services"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  make docs         - Build documentation"
	@echo "  make docs-serve   - Serve documentation locally"
	@echo ""
	@echo "$(YELLOW)Database:$(NC)"
	@echo "  make db-init      - Initialize database"
	@echo "  make db-migrate   - Run database migrations"
	@echo "  make db-reset     - Reset database"
	@echo ""
	@echo "$(YELLOW)Deployment:$(NC)"
	@echo "  make deploy-dev   - Deploy to development"
	@echo "  make deploy-prod  - Deploy to production"
	@echo "  make release      - Create new release"

# ===== DEVELOPMENT =====

install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev:
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	black .
	isort .
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint:
	@echo "$(YELLOW)Running linters...$(NC)"
	flake8 . --config=.flake8
	pylint *.py
	@echo "$(GREEN)✓ Linting complete$(NC)"

type-check:
	@echo "$(YELLOW)Running type checker...$(NC)"
	mypy . --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# ===== TESTING =====

test:
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v

test-cov:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "$(YELLOW)Coverage report: htmlcov/index.html$(NC)"

test-fast:
	@echo "$(GREEN)Running fast tests...$(NC)"
	pytest tests/ -v -m "not slow"

benchmark:
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(PYTHON) benchmarking.py
	@echo "$(YELLOW)Results saved to: benchmarks/$(NC)"

# ===== RUNNING =====

run:
	@echo "$(GREEN)Starting interactive demo...$(NC)"
	$(PYTHON) main.py

cli:
	@echo "$(GREEN)Starting CLI interface...$(NC)"
	$(PYTHON) cli.py --help

web:
	@echo "$(GREEN)Starting web interface on http://localhost:$(PORT)$(NC)"
	$(PYTHON) -m streamlit run web_interface.py --server.port=$(PORT)

api:
	@echo "$(GREEN)Starting API server on http://localhost:8000$(NC)"
	uvicorn api:app --reload --port=8000

worker:
	@echo "$(GREEN)Starting background worker service$(NC)"
	$(PYTHON) worker.py

jupyter:
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	jupyter notebook

# ===== DOCKER =====

docker-build:
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE)$(NC)"

docker-run:
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -it --rm -p $(PORT):$(PORT) -p 8000:8000 $(DOCKER_IMAGE)

docker-up:
	@echo "$(GREEN)Starting all services with docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "  Web UI: http://localhost:8501"
	@echo "  API: http://localhost:8000"
	@echo "  Jupyter: http://localhost:8888"

docker-down:
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs:
	docker-compose logs -f

# ===== DOCUMENTATION =====

docs:
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && $(MAKE) html
	@echo "$(GREEN)✓ Documentation built: docs/_build/html/index.html$(NC)"

docs-serve:
	@echo "$(GREEN)Serving documentation on http://localhost:8080$(NC)"
	cd docs/_build/html && python -m http.server 8080

# ===== DATABASE =====

db-init:
	@echo "$(GREEN)Initializing database...$(NC)"
	$(PYTHON) -c "from database import init_database; init_database()"
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate:
	@echo "$(GREEN)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations complete$(NC)"

db-reset:
	@echo "$(RED)Resetting database...$(NC)"
	@echo "$(RED)WARNING: This will delete all data! Press Ctrl+C to cancel$(NC)"
	@sleep 3
	rm -f negotiation_simulator.db
	$(MAKE) db-init
	@echo "$(GREEN)✓ Database reset$(NC)"

# ===== DEPLOYMENT =====

deploy-dev:
	@echo "$(YELLOW)Deploying to development environment...$(NC)"
	git push heroku develop:main
	@echo "$(GREEN)✓ Deployed to development$(NC)"

deploy-prod:
	@echo "$(RED)Deploying to production environment...$(NC)"
	@echo "$(RED)WARNING: Deploying to production! Press Ctrl+C to cancel$(NC)"
	@sleep 3
	git push production main
	@echo "$(GREEN)✓ Deployed to production$(NC)"

release:
	@echo "$(GREEN)Creating new release...$(NC)"
	@read -p "Enter version number (e.g., 1.0.0): " version; \
	git tag -a v$$version -m "Release v$$version"; \
	git push origin v$$version; \
	echo "$(GREEN)✓ Release v$$version created$(NC)"

# ===== UTILITIES =====

check-all: lint type-check test
	@echo "$(GREEN)✓ All checks passed$(NC)"

setup: install-dev db-init
	@echo "$(GREEN)✓ Development environment setup complete$(NC)"

demo-simple:
	@echo "$(GREEN)Running simple buyer-seller demo...$(NC)"
	$(PYTHON) -c "from main import quick_start; quick_start()"

demo-complex:
	@echo "$(GREEN)Running complex multi-party demo...$(NC)"
	$(PYTHON) cli.py run examples/climate_accord.yaml --viz

stats:
	@echo "$(YELLOW)Project Statistics:$(NC)"
	@echo "Lines of code: $$(find . -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "Python files: $$(find . -name '*.py' | wc -l)"
	@echo "Test files: $$(find tests -name '*.py' | wc -l)"
	@echo "Test cases: $$(grep -r "def test_" tests | wc -l)"

# ===== CI/CD =====

ci: check-all test-cov
	@echo "$(GREEN)✓ CI checks complete$(NC)"

pre-commit: format lint type-check test-fast
	@echo "$(GREEN)✓ Pre-commit checks complete$(NC)"

# ===== SHORTCUTS =====

i: install
id: install-dev
c: clean
f: format
l: lint
t: test
tc: test-cov
r: run
w: web
a: api
d: docker-build
du: docker-up
dd: docker-down

.DEFAULT_GOAL := help
