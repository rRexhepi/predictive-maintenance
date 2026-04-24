# predictive-maintenance

A FastAPI service that predicts the **Remaining Useful Life (RUL)** of
turbofan engines from sensor telemetry. Training uses the NASA **C-MAPSS**
degradation dataset; serving is a standalone Python API with single and
batch prediction endpoints.

> **Status:** portfolio project. Training + preprocessing + serving work
> end-to-end once the dataset is in `data/`. Rolling cleanup underway —
> see [Roadmap](#roadmap).

## Architecture

```mermaid
flowchart LR
    subgraph data["Data"]
        RAW["data/train_FD001.txt<br/>(NASA C-MAPSS)"]
    end
    subgraph prep["Preprocessing"]
        PP["src/preprocessing.py<br/>(handle outliers, RUL, lag feats, scale)"]
        PP --> CLEAN["data/{train,val}_preprocessed.csv"]
        PP --> SCALER["models/scaler.pkl"]
    end
    subgraph train["Training"]
        TM["src/train_model.py<br/>(RandomForest + MLflow)"]
        TM --> MODEL["models/rf_model.pkl"]
        TM --> MLFLOW[("MLflow<br/>mlruns/")]
    end
    subgraph serve["Serving"]
        API["FastAPI<br/>/predict · /batch_predict"]
        MODEL --> API
        SCALER --> API
        CLIENT["HTTP POST<br/>(access_token header)"] --> API
    end
    RAW --> PP
    CLEAN --> TM
```

## What's in the box

| Layer | Tech |
|---|---|
| API | FastAPI 0.115 + Pydantic v2 |
| Auth | API key via `access_token` header |
| Model | `RandomForestRegressor` (scikit-learn 1.5) |
| Tracking | MLflow 2.17 |
| Packaging | `requirements.txt` / `requirements-dev.txt` |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/rRexhepi/predictive-maintenance.git
cd predictive-maintenance
python3 -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements-dev.txt

# 2. Drop the C-MAPSS dataset into data/
#    See data/README.md for links.

# 3. Preprocess + fit the scaler
python src/preprocessing.py

# 4. Train the model (logs the run via MLflow; see below)
python src/train_model.py

# 5. Serve
export API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
uvicorn src.api.main:app --reload

# 6. Try it
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "access_token: $API_KEY" \
  -d '{"feature1": 0.5, "feature2": 1.2, "feature3": 3.4}'
```

FastAPI exposes Swagger UI at <http://127.0.0.1:8000/docs> and ReDoc at
<http://127.0.0.1:8000/redoc>.

## Docker

```bash
# Train artifacts first (the image mounts ./models read-only)
make preprocess && make train

export API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
make docker-build
make docker-up

curl -fsS http://localhost:8000/health
```

`docker-compose.yaml` mounts `./models:/app/models:ro` so a retrain
outside the container is picked up on the next restart without rebuilding
the image. `API_KEY` is required — the compose file fails fast if it's
unset rather than letting the API boot with no auth.

## MLflow Model Registry

`src/train_model.py` wraps training in `mlflow.start_run()`, logs a
single **pyfunc** artifact (scaler + estimator wrapped as
`PredictiveMaintenanceModel`), and registers it under the name
`predictive-maintenance-rul`. Every new version gets the `@candidate`
alias; passing `--promote` also moves the `@production` alias.

```bash
python src/train_model.py             # register as @candidate
python src/train_model.py --promote   # also promote to @production
mlflow ui                              # browse runs + registry at http://localhost:5000
```

Serving resolves either path:

```bash
# Registry path (production).
MODEL_URI=models:/predictive-maintenance-rul@production make serve

# Filesystem path (local dev / CI / Dockerfile default).
make serve   # falls back to models/rf_model.pkl + models/scaler.pkl
```

`MLFLOW_TRACKING_URI` defaults to the file backend `file:./mlruns` — no
long-running tracking server required for local iteration. Point it at
an HTTP backend (`http://localhost:5001` etc.) when you want the UI.
`PM_N_JOBS` caps sklearn's joblib fan-out (default 2; crank higher if
you have the cores).

## Metrics & drift

`GET /metrics` returns Prometheus text with per-call prediction
counters, a prediction-latency histogram, a predicted-RUL distribution
histogram, and PSI-based feature drift gauges computed from a rolling
buffer of recent inputs vs. the training distribution captured in
`models/reference_stats.json`.

```
# HELP pm_predictions_total Predictions served.
# TYPE pm_predictions_total counter
pm_predictions_total 42.0

# HELP pm_prediction_latency_seconds End-to-end latency of a /predict or /batch_predict call.
# TYPE pm_prediction_latency_seconds histogram
pm_prediction_latency_seconds_bucket{le="0.01"} 57.0
pm_prediction_latency_seconds_bucket{le="+Inf"} 60.0

# HELP pm_feature_drift_psi PSI of recent inputs vs. the training distribution.
# TYPE pm_feature_drift_psi gauge
pm_feature_drift_psi{feature="feature1"} 0.04
pm_feature_drift_psi{feature="feature2"} 0.12
```

**Why PSI not Evidently?** A `/metrics` endpoint backing a Grafana
dashboard wants a single scalar per feature to plot, which is exactly
what PSI gives. ~20 lines of `compute_psi` keeps the dep surface small
and documents what "drift" actually means. On a static dataset the
numbers sit near zero — the point is the wiring; point it at live
telemetry and it earns its keep. Threshold convention: `PSI < 0.1` =
no drift, `0.1–0.25` = moderate, `≥ 0.25` = significant.

### Observability stack

```bash
make docker-up    # api + prometheus + grafana
make dashboards   # prints URLs
```

Prometheus scrapes the API's `/metrics` directly on a 15s interval —
no Pushgateway needed (the API is long-lived). Grafana auto-loads
[`observability/grafana/dashboards/predictive-maintenance.json`](observability/grafana/dashboards/predictive-maintenance.json)
with:

* **Requests per minute** and **mean predicted RUL** at a glance.
* **p50 / p95 / p99 latency** computed from `histogram_quantile` over
  the prediction-latency histogram.
* **Max feature drift PSI** with green / yellow / red thresholds at
  0.0 / 0.1 / 0.25, plus a per-feature time series for drill-down.
* A table view of `pm_model_info` — registry URI or filesystem path +
  version — so on-call can tell at a glance which model is loaded.

**Alert rules.** Prometheus evaluates
[`observability/prometheus/alerts.yml`](observability/prometheus/alerts.yml):

* `PMAPIHighLatency` — p95 of `/predict` over 5m stays above 250ms for 10m.
* `PMFeatureDriftHigh` — any feature's PSI stays above 0.25 for 30m.
* `PMAPIDown` — scrape target missing for 5m.

Alertmanager isn't shipped yet — the rules expose their state on
Prometheus' `/alerts` page and via Grafana unified alerting.

![Predictive Maintenance — Serving Health & Drift dashboard](docs/grafana-dashboard.png)

The image above is captured in CI via the
[`Capture Grafana dashboard screenshot`](.github/workflows/screenshot-grafana-dashboard.yml)
workflow — an Ubuntu runner generates synthetic training data, brings
up the compose stack, fires a burst of `/predict` traffic, snapshots
the dashboard with Playwright, and opens a follow-up PR with the
refreshed PNG. Trigger it manually from the Actions tab whenever the
panels change.

## Endpoints

### `POST /predict`

Single-row prediction.

**Headers:** `Content-Type: application/json`, `access_token: <API_KEY>`

**Body:**
```json
{"feature1": 0.5, "feature2": 1.2, "feature3": 3.4}
```

**Response:** `{"predicted_rul": 15.2}`

### `POST /batch_predict`

Batch prediction.

**Body:**
```json
{"data": [
  {"feature1": 0.5, "feature2": 1.2, "feature3": 3.4},
  {"feature1": 0.6, "feature2": 1.3, "feature3": 3.5}
]}
```

**Response:** `{"predictions": [15.2, 23.5]}`

## Tests

```bash
make test
```

Covers:
- `Preprocessor`-style utility functions (drop/impute branches, no-mutation, file round-trip, `validate_data`, `save_predictions`).
- The FastAPI `/predict`, `/batch_predict`, and `/health` endpoints — happy paths, Pydantic 422s on bad payloads, 403 on missing / wrong API keys, empty-batch rejection, and a regression guard asserting no `temp_*.csv` files appear during a request.

CI runs ruff + pytest + a Docker image build on every push and pull request (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Known gaps (being worked on)

- **API schema is placeholder.** `PredictRequest` still has `feature1/2/3`
  fields; the actual trained model consumes the full C-MAPSS sensor +
  lag feature set. API and model are out of sync until the schema is
  regenerated from the training columns.

## Roadmap

What a reviewer would expect from a "production-style ML serving"
portfolio project, and what's next:

- [x] `requirements.txt` + `requirements-dev.txt`
- [x] Cleaned `.gitignore` (was globally ignoring `*.txt` and `*.yaml`, blocking itself)
- [x] `data/` and `models/` README pointers; artifacts gitignored
- [x] Untrack `mlflow.db` (was committed)
- [x] Kill disk-I/O in `/predict`; clean in-memory
- [x] Lifespan event handler (replacing deprecated `@app.on_event`)
- [x] Dockerfile + `docker-compose.yaml`
- [x] GitHub Actions CI: lint + pytest + Docker build
- [x] Endpoint tests (happy path + 403 on bad key + 422 on bad payload)
- [x] Move MLflow tracking URI to env var with file-store default
- [x] MLflow Model Registry with `@candidate`/`@production` aliases + pyfunc serving
- [x] Cap `n_jobs` default (now 2; env-configurable via `PM_N_JOBS`)
- [ ] Regenerate API schema from training feature columns
- [x] Prometheus `/metrics` endpoint (predictions counter + latency histogram + PSI drift gauges + RUL distribution)
- [x] Provisioned Grafana dashboard + alert rules
- [x] CI workflow that captures the dashboard screenshot via Playwright

## License

MIT — see [LICENSE](LICENSE).
