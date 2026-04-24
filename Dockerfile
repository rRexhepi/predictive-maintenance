FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is only needed for the healthcheck; nothing compiles on top of it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the source + the model artifacts the API serves. Artifacts go in
# via volume mount in docker-compose.yaml for local dev; bundling them
# here keeps the image self-sufficient when the compose file isn't used.
COPY src/ ./src/
COPY models/ ./models/

# The API looks up artifacts by env var; these are the defaults the
# Dockerfile ships with. Override via `-e` / compose to point at a
# different path, or set `MODEL_URI=models:/<name>@<alias>` to load
# from the MLflow Model Registry instead.
ENV PYTHONPATH=/app \
    MODEL_PATH=/app/models/rf_model.pkl \
    SCALER_PATH=/app/models/scaler.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
