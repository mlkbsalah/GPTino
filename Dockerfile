FROM python:3.14.10-slim-trixie

WORKDIR /workdir

RUN pip install --no-cache-dir uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
