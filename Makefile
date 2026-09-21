# mixengine — common tasks.
#
# `make` with no target prints this list. Anything that needs to be run
# more than once belongs here rather than in a paragraph of the README,
# because a command in prose drifts from the one CI actually runs.

PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
PORT    ?= 8000
DATA    ?= ./data

.DEFAULT_GOAL := help
.PHONY: help venv install install-all install-models test test-fast lint typecheck fmt \
        browser-test check-all \
        check serve doctor clean dist docker

help:  ## Show this list
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: venv  ## Install the engine and its dev tools
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[web,dev]"

install-all: venv  ## Install everything optional as well
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[quality,stretch,web,dev]"

test:  ## Run the full test suite
	$(BIN)/python -m pytest

test-fast:  ## Run only the tests that need no audio backend
	$(BIN)/python -m pytest -k "not render and not restoration"

lint:  ## Check style and common errors
	$(BIN)/ruff check src tests

fmt:  ## Fix what can be fixed automatically
	$(BIN)/ruff check --fix src tests

typecheck:  ## Run mypy
	$(BIN)/mypy

check: lint typecheck test  ## Everything CI runs

browser-test:  ## Run the worklet DSP checks headlessly (needs playwright + chromium)
	$(BIN)/python scripts/run_browser_tests.py

check-all: check browser-test  ## Both CI jobs

install-models: venv  ## torch-backed backends: pitch (torchcrepe), stems (demucs), beats (beat_this)
	$(BIN)/pip install "torch==2.7.*" "torchaudio==2.7.*"
	$(BIN)/pip install -e ".[models]"
	$(BIN)/pip install playwright && $(BIN)/playwright install chromium

serve:  ## Start the local interface
	MIXENGINE_DATA=$(DATA) $(BIN)/mixengine serve --port $(PORT)

doctor:  ## Report which optional components are installed
	$(BIN)/mixengine doctor

dist:  ## Build a wheel and an sdist
	$(BIN)/pip install -q build
	$(BIN)/python -m build

docker:  ## Build the container image
	docker build -t mixengine:2.0.0 .

clean:  ## Remove build artefacts and caches
	rm -rf build dist src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find src tests -name __pycache__ -type d -exec rm -rf {} +
