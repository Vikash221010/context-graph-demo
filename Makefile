.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend \
        generate-data clean clean-data test lint format \
        docker-up docker-down docker-logs health stats

# Default target
help:
	@echo "Context Graph Demo - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies (backend + frontend)"
	@echo "  make install-backend  Install backend dependencies with uv"
	@echo "  make install-frontend Install frontend dependencies with npm"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start both backend and frontend"
	@echo "  make dev-backend      Start backend server only"
	@echo "  make dev-frontend     Start frontend server only"
	@echo ""
	@echo "Data:"
	@echo "  make generate-data    Generate sample data in Neo4j"
	@echo "  make clean-data       Clear all data from Neo4j"
	@echo ""
	@echo "Docker (Local Neo4j):"
	@echo "  make docker-up        Start Neo4j with Docker Compose"
	@echo "  make docker-down      Stop Neo4j Docker container"
	@echo "  make docker-logs      View Neo4j Docker logs"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             Run backend tests"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code"
	@echo ""
	@echo "Utilities:"
	@echo "  make health           Check backend health"
	@echo "  make stats            Show Neo4j graph statistics"
	@echo "  make clean            Clean build artifacts"

# =============================================================================
# Setup
# =============================================================================

install: install-backend install-frontend
	@echo "All dependencies installed!"

install-backend:
	cd backend && uv venv && uv pip install -e .

install-frontend:
	cd frontend && npm install

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Starting backend and frontend..."
	@make dev-backend &
	@sleep 3
	@make dev-frontend

dev-backend:
	@echo "Starting backend on http://localhost:8000..."
	cd backend && source .venv/bin/activate && \
		export $$(grep -v '^#' ../.env | xargs) && \
		uvicorn app.main:app --reload --port 8000

dev-frontend:
	@echo "Starting frontend on http://localhost:3000..."
	cd frontend && npm run dev

# =============================================================================
# Data Management
# =============================================================================

generate-data:
	@echo "Generating sample data in Neo4j..."
	cd backend && source .venv/bin/activate && \
		export $$(grep -v '^#' ../.env | xargs) && \
		python scripts/generate_sample_data.py

clean-data:
	@echo "Clearing all data from Neo4j..."
	cd backend && source .venv/bin/activate && \
		export $$(grep -v '^#' ../.env | xargs) && \
		python -c "from app.context_graph_client import context_graph_client; \
			context_graph_client.driver.session().run('MATCH (n) DETACH DELETE n'); \
			print('All data cleared.')"

# =============================================================================
# Docker (Local Neo4j)
# =============================================================================

docker-up:
	docker-compose up -d
	@echo "Neo4j starting at http://localhost:7474"
	@echo "Waiting for Neo4j to be ready..."
	@sleep 10
	@echo "Neo4j should be ready now."

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f neo4j

# =============================================================================
# Testing & Quality
# =============================================================================

test:
	cd backend && source .venv/bin/activate && pytest

lint:
	cd backend && source .venv/bin/activate && ruff check .
	cd frontend && npm run lint

format:
	cd backend && source .venv/bin/activate && ruff format .
	cd frontend && npm run lint -- --fix

# =============================================================================
# Utilities
# =============================================================================

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

stats:
	@curl -s http://localhost:8000/api/graph/statistics | python3 -m json.tool

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.venv 2>/dev/null || true
	rm -rf frontend/node_modules 2>/dev/null || true
	rm -rf frontend/.next 2>/dev/null || true
	@echo "Cleaned build artifacts."
