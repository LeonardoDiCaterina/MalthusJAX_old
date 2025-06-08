# File: MalthusJAX/Makefile
.PHONY: help install install-dev test test-cov lint format clean setup-dev
.DEFAULT_GOAL := help

help: ## Show this help message
    @echo "MalthusJAX Development Commands:"
    @echo "================================"
    @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
    pip install -e .

install-dev: ## Install package with development dependencies  
    pip install -e ".[dev]"

install-all: ## Install package with all dependencies
    pip install -e ".[all]"

test: ## Run all tests
    pytest

test-cov: ## Run tests with coverage report
    pytest --cov=src/malthusjax --cov-report=html --cov-report=term-missing

lint: ## Run all linting checks
    @echo "Running flake8..."
    flake8 src tests
    @echo "Running black check..."
    black --check src tests  
    @echo "Running isort check..."
    isort --check-only src tests

format: ## Format code with black and isort
    @echo "Formatting with black..."
    black src tests
    @echo "Sorting imports with isort..."
    isort src tests

type-check: ## Run type checking with mypy
    mypy src/malthusjax

clean: ## Remove build artifacts and cache
    rm -rf build/ dist/ *.egg-info/
    rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
    find . -type d -name __pycache__ -delete
    find . -type f -name "*.pyc" -delete

setup-dev: install-dev ## Setup complete development environment
    @echo "🎉 Development environment setup complete!"
    @echo "💡 Try: make test, make format, make lint"

check-all: lint type-check test ## Run all quality checks and tests
    @echo "✅ All checks passed!"