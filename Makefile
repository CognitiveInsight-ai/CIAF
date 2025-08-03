# CIAF Development Makefile

.PHONY: help install install-dev test test-unit test-integration clean lint format docs build

# Default target
help:
	@echo "CIAF Development Commands"
	@echo "========================"
	@echo "install       - Install CIAF in development mode"
	@echo "install-dev   - Install development dependencies"
	@echo "test          - Run all tests"
	@echo "test-unit     - Run unit tests only"
	@echo "test-integration - Run integration tests only"
	@echo "lint          - Run code linting"
	@echo "format        - Format code with black and isort"
	@echo "docs          - Build documentation"
	@echo "clean         - Clean build artifacts"
	@echo "build         - Build distribution packages"

# Installation
install:
	python -m pip install -e .

install-dev:
	python -m pip install -e .
	python -m pip install -r requirements-dev.txt

# Testing
test:
	python scripts/run_tests.py all

test-unit:
	python scripts/run_tests.py unit

test-integration:
	python scripts/run_tests.py integration

# Code quality
lint:
	flake8 ciaf tests examples
	mypy ciaf

format:
	black ciaf tests examples scripts tools
	isort ciaf tests examples scripts tools

# Documentation
docs:
	cd docs && make html

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Build
build: clean
	python -m build

# Setup development environment
setup-dev:
	python scripts/setup_dev_env.py
