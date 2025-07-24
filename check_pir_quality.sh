#!/bin/bash
# PIR Module Linter Checks

echo "🔍 Running PIR Module Code Quality Checks..."
echo "============================================"

# Change to project directory
cd /Users/rohanvinaik/genomevault

# Run Black formatter
echo -e "\n📝 Running Black formatter..."
black genomevault/pir/ tests/pir/ scripts/bench_pir.py

# Run flake8
echo -e "\n🔎 Running flake8..."
flake8 genomevault/pir/ tests/pir/ scripts/bench_pir.py --config=.flake8

# Run mypy
echo -e "\n🔍 Running mypy type checker..."
mypy genomevault/pir/ --config-file=mypy.ini

# Run pylint
echo -e "\n📊 Running pylint..."
pylint genomevault/pir/ --rcfile=.pylintrc --exit-zero

# Run tests
echo -e "\n🧪 Running PIR tests..."
pytest tests/pir/ -v --tb=short

echo -e "\n✅ Code quality checks complete!"
