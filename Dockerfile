FROM python:3.14.10-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean

WORKDIR /workdir

RUN pip install --no-cache-dir uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
