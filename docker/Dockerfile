# Image de base Python legere
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app



COPY pyproject.toml uv.lock ./


RUN uv sync --frozen --no-dev

# uv installe dans /app/.venv : on l'ajoute au PATH pour que uvicorn/streamlit
# soient directement executables au runtime.
ENV PATH="/app/.venv/bin:$PATH"


COPY src/ ./src/
COPY models/ ./models/

# Port expose par l'API
EXPOSE 8000



CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]