.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --group dev

.PHONY: run
run:
	uv run python test.py

.PHONY: format
format:
	uv run black .
	uv run ruff check --fix .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache

.PHONY: clean-data
clean-data:
	rm -rf data/

.PHONY: clean-all
clean-all: clean clean-data
	rm -rf mlruns/ models/ results/

.PHONY: sweep
sweep:
	uv run python experiments/sweep.py --sweep experiments/configs/mnist_sweep.yaml

.PHONY: shell
shell:
	uv run ipython

# ── Pi Agent Extensions ──────────────────────────────────────────────

.PHONY: pi
pi:
	pi -e extensions/footer.ts -e extensions/experiment-enforcer.ts

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  install-dev  - Install with dev dependencies"
	@echo "  run          - Run test.py"
	@echo "  format       - Format code with black and ruff"
	@echo "  lint         - Lint code with ruff"
	@echo "  clean        - Remove cache files"
	@echo "  shell        - Start IPython shell"
