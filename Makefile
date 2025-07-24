.PHONY: format check fix lint test install-dev help quality tailchasing

help:
	@echo "GenomeVault Development Commands"
	@echo "================================"
	@echo "format      - Format code with Black and isort"
	@echo "check       - Check code quality without fixing"
	@echo "fix         - Auto-fix common code issues"
	@echo "lint        - Run comprehensive linting"
	@echo "test        - Run test suite"
	@echo "install-dev - Install development dependencies"
	@echo "quality     - Run full quality analysis"
	@echo "tailchasing - Check for LLM anti-patterns"

format:
	python -m black .
	python -m isort .

check:
	python -m black --check .
	python -m isort --check-only .
	python -m flake8 .

fix:
	python -m autoflake --remove-unused-variables --remove-all-unused-imports --in-place --recursive .
	python -m black .
	python -m isort .

lint:
	python -m pylint genomevault/ clinical_validation/ --exit-zero

test:
	python -m pytest tests/ -v

install-dev:
	pip install black isort flake8 pylint autoflake pre-commit pytest

quality:
	python comprehensive_code_quality_fixer.py

tailchasing:
	python -m tailchasing . --show-suggestions

report:
	python final_code_quality_report.py
