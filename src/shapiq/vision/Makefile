# Vision package checks and tests.
#
# Usage (from anywhere):
#   make -C src/shapiq/vision check
#   make -C src/shapiq/vision test

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
VISION_SRC := src/shapiq/vision
VISION_TESTS := tests/shapiq/tests_unit/tests_vision
VISION_PY := $(wildcard $(ROOT)/$(VISION_SRC)/*.py)
RUFF_PATHS := $(VISION_PY) $(ROOT)/$(VISION_TESTS)
PYTEST := uv run pytest
RUFF := uv run ruff

.PHONY: help check lint format fix test test-cov

help:
	@echo "Vision package targets:"
	@echo "  check     Run lint, format check, and tests"
	@echo "  lint      Run ruff linter on vision *.py and tests"
	@echo "  format    Check ruff formatting (no writes)"
	@echo "  fix       Auto-fix lint and format issues"
	@echo "  test      Run all vision unit tests"
	@echo "  test-cov  Run vision tests with coverage for shapiq.vision"

check: lint format test

lint:
	cd $(ROOT) && $(RUFF) check $(RUFF_PATHS)

format:
	cd $(ROOT) && $(RUFF) format --check $(RUFF_PATHS)

fix:
	cd $(ROOT) && $(RUFF) check --fix $(RUFF_PATHS)
	cd $(ROOT) && $(RUFF) format $(RUFF_PATHS)

test:
	cd $(ROOT) && $(PYTEST) $(VISION_TESTS)

test-cov:
	cd $(ROOT) && $(PYTEST) $(VISION_TESTS) --cov=shapiq.vision --cov-report=term-missing
