.PHONY: help install install-dev lint fmt test preprocess train serve docker-build docker-up docker-down clean

help:
	@echo "Targets:"
	@echo "  install      - Install runtime deps"
	@echo "  install-dev  - Install runtime + dev deps (pytest, ruff)"
	@echo "  lint         - Run ruff"
	@echo "  fmt          - Auto-fix lint + format"
	@echo "  test         - Run pytest"
	@echo "  preprocess   - Run the data preprocessing pipeline"
	@echo "  train        - Train the model (MLflow tracking)"
	@echo "  serve        - Run the FastAPI app locally on :8000"
	@echo "  docker-build - Build the API image"
	@echo "  docker-up    - docker compose up -d (api + prometheus + grafana)"
	@echo "  docker-down  - docker compose down"
	@echo "  dashboards   - Print URLs for API / Grafana / Prometheus"
	@echo "  clean        - Remove __pycache__, logs, caches"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

fmt:
	ruff check --fix .
	ruff format .

test:
	pytest

preprocess:
	python src/preprocessing.py

train:
	python src/train_model.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t predictive-maintenance-api .

docker-up:
	@test -n "$$API_KEY" || (echo "ERROR: export API_KEY before docker-up" && exit 1)
	docker compose up -d --build

docker-down:
	docker compose down

dashboards:
	@echo "API        : http://localhost:8000  (FastAPI docs at /docs, Prom metrics at /metrics)"
	@echo "Grafana    : http://localhost:3000  (anonymous viewer enabled — dashboard 'Predictive Maintenance API — Serving Health & Drift')"
	@echo "Prometheus : http://localhost:9090"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache prediction.log
