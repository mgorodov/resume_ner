.PHONY: run
run:
	uv run python3 -m app.cmd

.PHONY: format
format:
	uv run ruff format
	uv run ruff check . --fix

.PHONY: lint
lint:
	uv run ruff check .
	uv run mypy .
