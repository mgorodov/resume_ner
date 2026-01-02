FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_NO_DEV=1
RUN uv sync --locked --no-install-project

COPY certs ./certs
COPY .env .
COPY app ./app

RUN uv sync --locked
CMD uv run python3 -m app.cmd
