# Vision package checks and tests.
#
# Usage (from anywhere):
#   make -C src/shapiq/vision check
#   make -C src/shapiq/vision test
#
# Code quality uses the same pre-commit hooks as .github/workflows/ci.yml,
# scoped to vision library code, tests, and this Makefile.
# The ty hook in pre-commit checks all of src/shapiq, use `make typecheck` for vision only.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
VISION_SRC := src/shapiq/vision
VISION_TESTS := tests/shapiq/tests_unit/tests_vision
VISION_PY := $(wildcard $(ROOT)/$(VISION_SRC)/*.py)
VISION_QUALITY_FILES := $(shell git -C $(ROOT) ls-files \
	$(VISION_SRC)/*.py \
	$(VISION_SRC)/Makefile \
	$(VISION_TESTS)/)
# Pre-commit hooks that honor --files (mirrors .pre-commit-config.yaml minus ty).
PRE_COMMIT_SKIP_TY := SKIP=ty
PYTEST := uv run pytest
PRE_COMMIT := uv run pre-commit

.PHONY: help check quality fix test test-cov typecheck

help:
	@echo "Vision package targets:"
	@echo "  check      Run pre-commit, ty, and unit tests"
	@echo "  quality    Run pre-commit hooks on tracked vision files"
	@echo "  fix        Run pre-commit hooks and apply auto-fixes"
	@echo "  typecheck  Run ty on vision library modules"
	@echo "  test       Run all vision unit tests"
	@echo "  test-cov   Run vision tests with coverage for shapiq.vision"

check: quality typecheck test

quality:
	cd $(ROOT) && $(PRE_COMMIT_SKIP_TY) $(PRE_COMMIT) run \
		--files $(VISION_QUALITY_FILES) --show-diff-on-failure

fix:
	cd $(ROOT) && $(PRE_COMMIT_SKIP_TY) $(PRE_COMMIT) run --files $(VISION_QUALITY_FILES)

typecheck:
	cd $(ROOT) && uv run ty check $(VISION_PY)

test:
	cd $(ROOT) && $(PYTEST) $(VISION_TESTS)

test-cov:
	cd $(ROOT) && $(PYTEST) $(VISION_TESTS) --cov=shapiq.vision --cov-report=term-missing
