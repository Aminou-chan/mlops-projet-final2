# Image de base Python legere
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app


COPY pyproject.toml uv.lock ./


RUN uv sync --frozen --no-dev


COPY src/ ./src/
COPY models/ ./models/

# Port expose par l'API
EXPOSE 8000



CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]