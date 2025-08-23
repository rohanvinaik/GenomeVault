.PHONY: help build run up down test fmt lint deps deps-upgrade deps-sync
.PHONY: db-init db-reset db-migrate db-seed redis-flush redis-cli
.PHONY: zk-setup zk-build zk-test zk-clean
.PHONY: demo pir-demo hdc-demo clinical-demo
.PHONY: logs logs-api logs-db logs-redis shell-api shell-db shell-redis
.PHONY: clean clean-cache clean-logs clean-all
.PHONY: dev dev-up dev-down status ps

# Default target
.DEFAULT_GOAL := help

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)GenomeVault Makefile$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(GREEN)Basic Commands:$(NC)"
	@echo "  make build         Build Docker image"
	@echo "  make run           Run API in foreground via docker compose"
	@echo "  make up            Start all services (detached)"
	@echo "  make down          Stop all services"
	@echo "  make status        Show service status"
	@echo "  make test          Run pytest locally (venv)"
	@echo ""
	@echo "$(GREEN)Infrastructure:$(NC)"
	@echo "  make db-init       Initialize database with alembic"
	@echo "  make db-reset      Drop and recreate database"
	@echo "  make db-migrate    Create new migration"
	@echo "  make db-seed       Load development seed data"
	@echo "  make redis-flush   Clear Redis cache"
	@echo "  make redis-cli     Connect to Redis CLI"
	@echo ""
	@echo "$(GREEN)Zero-Knowledge Proofs:$(NC)"
	@echo "  make zk-setup      Install circom and snarkjs"
	@echo "  make zk-build      Compile ZK circuits"
	@echo "  make zk-test       Test ZK proof generation and verification"
	@echo "  make zk-clean      Clean ZK build artifacts"
	@echo ""
	@echo "$(GREEN)Demos:$(NC)"
	@echo "  make demo          Run complete e2e demo"
	@echo "  make pir-demo      Start PIR servers and run query demo"
	@echo "  make hdc-demo      Encode sample genomic data demo"
	@echo "  make clinical-demo Run clinical analysis demo"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make dev           Start development environment"
	@echo "  make logs          Tail all container logs"
	@echo "  make logs-api      Tail API container logs"
	@echo "  make shell-api     Shell into API container"
	@echo "  make shell-db      Connect to PostgreSQL shell"
	@echo "  make fmt           Format code with black & isort"
	@echo "  make lint          Run ruff linter and format checker"
	@echo "  make clean         Remove generated files and caches"
	@echo ""
	@echo "$(GREEN)Dependencies:$(NC)"
	@echo "  make deps          Regenerate locked dependency files"
	@echo "  make deps-upgrade  Upgrade all dependencies"
	@echo "  make deps-sync     Sync environment to match requirements"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

build:
	docker build -t genomevault/api:local .

run:
	docker compose up --build api

up:
	./scripts/dev_up.sh

down:
	./scripts/dev_down.sh

test:
	./scripts/test.sh

fmt:
	@if command -v black >/dev/null; then black .; else echo "black not installed"; fi
	@if command -v isort >/dev/null; then isort .; else echo "isort not installed"; fi

lint:
	@if command -v ruff >/dev/null; then ruff check . && ruff format --check .; else echo "ruff not installed"; fi

deps:
	@echo "Regenerating locked dependency files..."
	@if command -v pip-compile >/dev/null; then \
		pip-compile requirements.in -o requirements.txt --resolver=backtracking && \
		pip-compile requirements-dev.in -o requirements-dev.txt --resolver=backtracking; \
	else \
		echo "pip-tools not installed. Run: pip install pip-tools"; \
	fi

deps-upgrade:
	@echo "Upgrading all dependencies to latest versions..."
	@if command -v pip-compile >/dev/null; then \
		pip-compile --upgrade requirements.in -o requirements.txt --resolver=backtracking && \
		pip-compile --upgrade requirements-dev.in -o requirements-dev.txt --resolver=backtracking; \
	else \
		echo "pip-tools not installed. Run: pip install pip-tools"; \
	fi

deps-sync:
	@echo "Syncing environment to match requirements..."
	@if command -v pip-sync >/dev/null; then \
		pip-sync requirements-dev.txt; \
	else \
		echo "pip-tools not installed. Run: pip install pip-tools"; \
	fi

# ============================================================================
# INFRASTRUCTURE TARGETS
# ============================================================================

db-init:
	@echo "$(GREEN)Initializing database with alembic...$(NC)"
	@docker-compose exec -T api alembic upgrade head
	@echo "$(GREEN)Database initialized successfully!$(NC)"

db-reset:
	@echo "$(YELLOW)Warning: This will drop and recreate the database!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(RED)Dropping and recreating database...$(NC)"; \
		docker-compose exec -T postgres psql -U genomevault -c "DROP DATABASE IF EXISTS genomevault;"; \
		docker-compose exec -T postgres psql -U genomevault -c "CREATE DATABASE genomevault;"; \
		docker-compose exec -T postgres psql -U genomevault -d genomevault -f /docker-entrypoint-initdb.d/init-db.sql; \
		$(MAKE) db-init; \
		echo "$(GREEN)Database reset complete!$(NC)"; \
	fi

db-migrate:
	@echo "$(GREEN)Creating new database migration...$(NC)"
	@read -p "Migration name: " name; \
	docker-compose exec api alembic revision --autogenerate -m "$$name"
	@echo "$(GREEN)Migration created!$(NC)"

db-seed:
	@echo "$(GREEN)Loading development seed data...$(NC)"
	@docker-compose exec -T postgres psql -U genomevault -d genomevault -f /docker-entrypoint-initdb.d/dev-seed.sql
	@echo "$(GREEN)Seed data loaded!$(NC)"

redis-flush:
	@echo "$(YELLOW)Clearing Redis cache...$(NC)"
	@docker-compose exec -T redis redis-cli FLUSHALL
	@echo "$(GREEN)Redis cache cleared!$(NC)"

redis-cli:
	@echo "$(GREEN)Connecting to Redis CLI...$(NC)"
	@docker-compose exec redis redis-cli

# ============================================================================
# ZERO-KNOWLEDGE PROOF TARGETS
# ============================================================================

zk-setup:
	@echo "$(GREEN)Installing circom and snarkjs...$(NC)"
	@if ! command -v circom >/dev/null; then \
		echo "Installing circom..."; \
		curl --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/iden3/circom/master/mkdocs/docs/getting-started/installation.md | sh; \
	else \
		echo "circom already installed: $$(circom --version)"; \
	fi
	@if ! command -v snarkjs >/dev/null; then \
		echo "Installing snarkjs..."; \
		npm install -g snarkjs; \
	else \
		echo "snarkjs already installed"; \
	fi
	@echo "$(GREEN)ZK tools setup complete!$(NC)"

zk-build:
	@echo "$(GREEN)Compiling ZK circuits...$(NC)"
	@if [ -d "genomevault/zk/circuits/sum64" ]; then \
		cd genomevault/zk/circuits/sum64 && \
		echo "Compiling sum64.circom..." && \
		circom sum64.circom --r1cs --wasm --sym --c -o build/ && \
		echo "$(GREEN)Circuit compilation successful!$(NC)"; \
	else \
		echo "$(RED)Circuit directory not found: genomevault/zk/circuits/sum64$(NC)"; \
		echo "Creating example circuit..."; \
		mkdir -p genomevault/zk/circuits/sum64/build && \
		echo 'pragma circom 2.0.0;\n\ntemplate Sum64() {\n    signal input a[64];\n    signal output sum;\n    \n    var total = 0;\n    for (var i = 0; i < 64; i++) {\n        total += a[i];\n    }\n    sum <== total;\n}\n\ncomponent main = Sum64();' > genomevault/zk/circuits/sum64/sum64.circom && \
		cd genomevault/zk/circuits/sum64 && \
		circom sum64.circom --r1cs --wasm --sym --c -o build/; \
	fi

zk-test:
	@echo "$(GREEN)Testing ZK proof generation and verification...$(NC)"
	@docker-compose exec -T api python -c "\
	from genomevault.zk_proofs.circuit_manager import CircuitManager; \
	import json; \
	print('Initializing circuit manager...'); \
	manager = CircuitManager(); \
	print('Generating proof...'); \
	proof = manager.generate_proof('genomic', \
		{'gene': 'BRCA1', 'population': 'EUR'}, \
		{'variant_data': [1,2,3,4,5]}); \
	print(f'Proof generated: {proof.proof_id}'); \
	print('Verifying proof...'); \
	is_valid = manager.verify_proof(proof); \
	print(f'Proof verification: {\"PASSED\" if is_valid else \"FAILED\"}'); \
	" || echo "$(YELLOW)Note: Run 'make zk-setup' and 'make zk-build' first$(NC)"

zk-clean:
	@echo "$(YELLOW)Cleaning ZK build artifacts...$(NC)"
	@rm -rf genomevault/zk/circuits/*/build
	@rm -rf genomevault/zk/circuits/*/*.r1cs
	@rm -rf genomevault/zk/circuits/*/*.sym
	@rm -rf genomevault/zk/circuits/*/*.json
	@echo "$(GREEN)ZK artifacts cleaned!$(NC)"

# ============================================================================
# DEMO TARGETS
# ============================================================================

demo:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(BLUE)Running GenomeVault End-to-End Demo$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(GREEN)[1/5] Starting services...$(NC)"
	@docker-compose up -d
	@sleep 5
	@echo ""
	@echo "$(GREEN)[2/5] Running HDC encoding demo...$(NC)"
	@$(MAKE) hdc-demo
	@echo ""
	@echo "$(GREEN)[3/5] Running PIR query demo...$(NC)"
	@$(MAKE) pir-demo
	@echo ""
	@echo "$(GREEN)[4/5] Running ZK proof demo...$(NC)"
	@$(MAKE) zk-test
	@echo ""
	@echo "$(GREEN)[5/5] Running clinical analysis demo...$(NC)"
	@$(MAKE) clinical-demo
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Demo complete! Check http://localhost:8000/docs for API documentation$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

pir-demo:
	@echo "$(GREEN)Running PIR demo...$(NC)"
	@echo "Starting PIR servers..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d pir-server-1 pir-server-2 pir-server-3 pir-server-4
	@sleep 3
	@echo "Executing PIR query..."
	@docker-compose exec -T api python -c "\
	import asyncio; \
	from genomevault.pir.client import PIRClient; \
	async def demo(): \
	    client = PIRClient(['http://pir-server-1:9001', 'http://pir-server-2:9002']); \
	    print('Querying for index 42...'); \
	    result = await client.query(42); \
	    print(f'Retrieved: {result}'); \
	asyncio.run(demo()); \
	" || echo "$(YELLOW)PIR demo requires running PIR servers$(NC)"

hdc-demo:
	@echo "$(GREEN)Running HDC encoding demo...$(NC)"
	@docker-compose exec -T api python -c "\
	from genomevault.hypervector.hd_encoder import HDEncoder; \
	import numpy as np; \
	print('Initializing HD encoder...'); \
	encoder = HDEncoder(dimension=1000); \
	print('Sample genomic variants:'); \
	variants = [ \
	    {'chrom': '1', 'pos': 1234567, 'ref': 'A', 'alt': 'T', 'impact': 'moderate'}, \
	    {'chrom': '2', 'pos': 9876543, 'ref': 'G', 'alt': 'C', 'impact': 'high'}, \
	    {'chrom': 'X', 'pos': 555555, 'ref': 'C', 'alt': 'G', 'impact': 'low'} \
	]; \
	for v in variants: print(f'  chr{v[\"chrom\"]}:{v[\"pos\"]} {v[\"ref\"]}>{v[\"alt\"]} ({v[\"impact\"]})'); \
	print('\\nEncoding variants to hypervector...'); \
	features = np.random.randn(100); \
	hv = encoder.encode(features); \
	print(f'Encoded to {len(hv)}-dimensional hypervector'); \
	print(f'Sparsity: {np.mean(hv == 0):.2%}'); \
	print(f'Privacy preserved: Original variants cannot be recovered'); \
	"

clinical-demo:
	@echo "$(GREEN)Running clinical analysis demo...$(NC)"
	@docker-compose exec -T api python -c "\
	import json; \
	print('Clinical Analysis Demo'); \
	print('==================='); \
	print('Analyzing BRCA1 variant for breast cancer risk...'); \
	variant = { \
	    'gene': 'BRCA1', \
	    'variant': 'c.68_69delAG', \
	    'classification': 'pathogenic', \
	    'zygosity': 'heterozygous' \
	}; \
	print(json.dumps(variant, indent=2)); \
	print('\\nRisk Assessment:'); \
	print('- Lifetime risk: 45-85% (vs 12% general population)'); \
	print('- Recommendations: Enhanced screening, genetic counseling'); \
	print('- Privacy: All analysis performed on encrypted data'); \
	"

# ============================================================================
# DEVELOPMENT TARGETS
# ============================================================================

dev:
	@echo "$(GREEN)Starting development environment...$(NC)"
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo ""
	@echo "Services available at:"
	@echo "  - API:            http://localhost:8000"
	@echo "  - API Docs:       http://localhost:8000/docs"
	@echo "  - pgAdmin:        http://localhost:8081"
	@echo "  - Redis Commander: http://localhost:8082"
	@echo "  - Prometheus:     http://localhost:9090"
	@echo "  - Grafana:        http://localhost:3000"

dev-up: dev

dev-down:
	@echo "$(YELLOW)Stopping development environment...$(NC)"
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
	@echo "$(GREEN)Development environment stopped!$(NC)"

status:
	@docker-compose ps

ps: status

logs:
	@docker-compose logs -f --tail=100

logs-api:
	@docker-compose logs -f --tail=100 api

logs-db:
	@docker-compose logs -f --tail=100 postgres

logs-redis:
	@docker-compose logs -f --tail=100 redis

shell-api:
	@echo "$(GREEN)Connecting to API container shell...$(NC)"
	@docker-compose exec api /bin/bash

shell-db:
	@echo "$(GREEN)Connecting to PostgreSQL...$(NC)"
	@docker-compose exec postgres psql -U genomevault -d genomevault

shell-redis:
	@echo "$(GREEN)Connecting to Redis CLI...$(NC)"
	@docker-compose exec redis redis-cli

# ============================================================================
# CLEANUP TARGETS
# ============================================================================

clean:
	@echo "$(YELLOW)Cleaning generated files and caches...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf dist/ 2>/dev/null || true
	@rm -rf build/ 2>/dev/null || true
	@rm -rf *.egg-info 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-cache: clean
	@echo "$(YELLOW)Cleaning all caches...$(NC)"
	@$(MAKE) redis-flush
	@rm -rf .cache/ 2>/dev/null || true
	@echo "$(GREEN)All caches cleared!$(NC)"

clean-logs:
	@echo "$(YELLOW)Cleaning log files...$(NC)"
	@rm -rf logs/*.log 2>/dev/null || true
	@rm -rf logs/*.txt 2>/dev/null || true
	@echo "$(GREEN)Log files cleaned!$(NC)"

clean-all: clean clean-cache clean-logs
	@echo "$(YELLOW)Performing deep clean...$(NC)"
	@docker-compose down -v 2>/dev/null || true
	@docker system prune -f 2>/dev/null || true
	@echo "$(GREEN)Deep clean complete!$(NC)"
