# Audio Restoration Pipeline - Full Phases 1-6
# Usage:
#   make help         # Show all targets
#   make install      # Install minimal dependencies
#   make install-ai   # Install AI extras
#   make run-core     # Run phases 1-2 only
#   make run-full     # Run all phases (no AI)
#   make run-ai       # Run all phases with AI
#   make dry-run      # Test pipeline without execution
#   make clean        # Remove outputs and logs

SHELL := /bin/bash
PY=python3
VENV?=.venv

# Default config files
CFG_CORE=configs/core_only.yaml
CFG_FULL=configs/full_no_ai.yaml
CFG_AI=configs/full_with_ai.yaml

# Default inputs
INPUT?=jock_itntvw.wav
NOISE?=evilhild_2.wav
REFERENCE?=

# Pipeline options
PHASES?=all
PROFILE?=streaming
STRICT?=false
DRY_RUN?=false

.PHONY: help install install-ai run-core run-full run-ai dry-run test clean

help:
	@echo "Audio Restoration Pipeline - Make Targets"
	@echo "========================================="
	@echo "Setup:"
	@echo "  make install      - Install minimal dependencies"
	@echo "  make install-ai   - Install AI extras (demucs, spleeter, etc.)"
	@echo ""
	@echo "Run Pipeline:"
	@echo "  make run-core     - Run phases 1-2 only (basic denoising)"
	@echo "  make run-full     - Run all phases except AI"
	@echo "  make run-ai       - Run all phases including AI (requires install-ai)"
	@echo ""
	@echo "Testing:"
	@echo "  make dry-run      - Test pipeline without execution"
	@echo "  make test         - Run unit tests"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Remove all outputs and logs"
	@echo "  make metrics      - Run metrics evaluation only"
	@echo "  make run-core-fast - Run minimal core DSP (no analysis)"
	@echo ""
	@echo "Options (set via command line):"
	@echo "  INPUT=<file>      - Input audio file (default: jock_itntvw.wav)"
	@echo "  NOISE=<file>      - Noise sample file (default: evilhild_2.wav)"
	@echo "  REFERENCE=<file>  - Reference clean audio for metrics"
	@echo "  PHASES=<list>     - Phases to run (e.g., 1,2,3 or all)"
	@echo "  PROFILE=<name>    - Mastering profile (streaming/podcast/broadcast)"
	@echo "  STRICT=true       - Fail on missing dependencies"
	@echo "  DRY_RUN=true      - Plan without execution"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pyyaml  # Ensure YAML support
	@echo "Minimal dependencies installed."
	@echo "For AI features, run: make install-ai"

install-ai:
	$(PY) -m pip install -r ai-extras.txt
	@echo "Installing Demucs (may take time)..."
	$(PY) -m pip install demucs
	@echo "Installing Spleeter..."
	$(PY) -m pip install spleeter
	@echo "AI dependencies installed."
	@echo "For GPU support, install PyTorch with CUDA separately."

run-core:
	$(PY) src/pipeline_v2.py --config $(CFG_CORE) \
		--input $(INPUT) --noise $(NOISE) \
		--phases 1,2

run-full:
	$(PY) src/pipeline_v2.py --config $(CFG_FULL) \
		--input $(INPUT) --noise $(NOISE) \
		--phases all --profile $(PROFILE)

run-ai:
	$(PY) src/pipeline_v2.py --config $(CFG_AI) \
		--input $(INPUT) --noise $(NOISE) \
		--phases all --profile $(PROFILE)

run-core-fast:
	$(PY) src/core_pipeline.py --config $(CFG_CORE) \
		--input $(INPUT) --noise $(NOISE)

run-custom:
	$(PY) src/pipeline_v2.py --config configs/full_pipeline.yaml \
		--input $(INPUT) --noise $(NOISE) \
		--phases $(PHASES) --profile $(PROFILE) \
		$(if $(filter true,$(STRICT)),--strict) \
		$(if $(filter true,$(DRY_RUN)),--dry-run)

dry-run:
	$(PY) src/pipeline_v2.py --config $(CFG_FULL) \
		--input $(INPUT) --noise $(NOISE) \
		--phases all --dry-run

metrics:
	$(PY) src/metrics_eval.py --config $(CFG_FULL) \
		$(if $(REFERENCE),--reference $(REFERENCE))

test:
	@echo "Running unit tests..."
	$(PY) -m pytest tests/ -v

clean:
	rm -rf outputs logs reports .cache __pycache__ src/__pycache__
	@echo "Cleaned outputs, logs, reports, and cache."

# Development targets
dev-setup:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt
	$(VENV)/bin/python -m pip install pytest black flake8
	@echo "Development environment ready in $(VENV)"

format:
	$(PY) -m black src/ tests/

lint:
	$(PY) -m flake8 src/ tests/ --max-line-length=100

# Quick test runs
quick-test:
	$(PY) src/pipeline_v2.py --config $(CFG_CORE) \
		--phases 1 --dry-run

benchmark:
	@echo "Benchmarking pipeline performance..."
	@time $(MAKE) run-core
	@echo "Core pipeline complete."