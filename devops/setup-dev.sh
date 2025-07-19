#!/bin/bash
# setup-dev.sh - Development environment setup for GenomeVault 3.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        exit 1
    fi
}

# Check prerequisites
log_info "Checking prerequisites..."
check_command python3
check_command pip
check_command docker
check_command git

# Python version check
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    log_error "Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Create project directory structure if needed
log_info "Setting up project structure..."
mkdir -p tests/{unit,integration,performance,security}
mkdir -p docs/test-reports

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
log_info "Installing project dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit
log_info "Installing pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create test database config
log_info "Creating test database configuration..."
cat > docker-compose.test.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: genomevault_test
      POSTGRES_USER: genomevault
      POSTGRES_PASSWORD: genomevault_test
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"

volumes:
  postgres_test_data:
EOF

# Setup test database
log_info "Setting up test database..."
docker-compose -f docker-compose.test.yml up -d postgres redis
sleep 5

# Run initial tests to verify setup
log_info "Running initial test suite..."
pytest tests/ -v --tb=short -m "not slow" || true

# Generate initial coverage report
log_info "Generating coverage report..."
pytest tests/ --cov=genomevault --cov-report=html --cov-report=term || true

# Setup git hooks
log_info "Setting up git hooks..."
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Run tests before push
echo "Running tests before push..."
pytest tests/ -x --tb=short -m "not slow"
EOF
chmod +x .git/hooks/pre-push

# Create test data fixtures
log_info "Creating test data fixtures..."
mkdir -p tests/fixtures
cat > tests/fixtures/sample_variants.json << 'EOF'
{
  "format": "vcf-4.3",
  "reference": "GRCh38",
  "samples": [{
    "id": "TEST001",
    "variants": [
      {"chr": "1", "pos": 14370, "ref": "G", "alt": "A", "rsid": "rs6054257"},
      {"chr": "2", "pos": 17330, "ref": "T", "alt": "C", "rsid": "rs1234567"}
    ]
  }]
}
EOF

# Create VS Code settings
log_info "Creating VS Code settings..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.path": "isort",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
EOF

# Generate test runner script
log_info "Creating test runner scripts..."
cat > run-tests.sh << 'EOF'
#!/bin/bash
# Test runner with various options

case "${1:-all}" in
  unit)
    pytest tests/unit/ -v
    ;;
  integration)
    pytest tests/integration/ -v
    ;;
  security)
    pytest tests/security/ -v
    bandit -r genomevault/
    ;;
  performance)
    pytest tests/performance/ -v --benchmark-only
    ;;
  coverage)
    pytest tests/ --cov=genomevault --cov-report=html --cov-report=term
    ;;
  ci)
    # Run all checks as in CI
    black --check .
    isort --check-only .
    flake8 .
    mypy genomevault
    pytest tests/ --cov=genomevault
    ;;
  watch)
    ptw tests/ -- -v
    ;;
  all)
    pytest tests/ -v
    ;;
  *)
    echo "Usage: $0 {unit|integration|security|performance|coverage|ci|watch|all}"
    exit 1
    ;;
esac
EOF
chmod +x run-tests.sh

# Create Makefile for common tasks
log_info "Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help test coverage lint format security clean docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make test        - Run all tests"
	@echo "  make coverage    - Run tests with coverage"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make security    - Run security checks"
	@echo "  make clean       - Clean up generated files"
	@echo "  make docker-up   - Start docker services"
	@echo "  make docker-down - Stop docker services"

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=genomevault --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

lint:
	black --check .
	isort --check-only .
	flake8 .
	mypy genomevault

format:
	black .
	isort .

security:
	bandit -r genomevault/
	safety check

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

docker-up:
	docker-compose -f docker-compose.test.yml up -d

docker-down:
	docker-compose -f docker-compose.test.yml down
EOF

log_info "Development environment setup complete!"
log_info "Next steps:"
echo "  1. Run 'make test' to run the test suite"
echo "  2. Run 'make coverage' to generate coverage report"
echo "  3. Run './run-tests.sh watch' for test watching"
echo "  4. Commit with 'git commit' to trigger pre-commit hooks"
