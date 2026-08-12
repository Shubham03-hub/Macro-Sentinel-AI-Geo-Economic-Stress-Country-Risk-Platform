# Deployment Guide

## 1. Local Deployment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# generate data (skip if you've dropped real CSVs into data/raw/)
python scripts/generate_synthetic_data.py

# run the full pipeline: preprocess -> features -> train -> evaluate -> predict
python main.py --stage all

# launch the dashboard
streamlit run dashboard/app.py
```
Dashboard will be available at `http://localhost:8501`.

To inspect experiment runs:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
MLflow UI will be available at `http://localhost:5000`.

## 2. Docker Deployment

Build and run the dashboard only (assumes `models/` and `data/processed/` already populated,
either from a local run or a mounted volume):
```bash
docker build -t macro-sentinel .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models macro-sentinel
```

Using docker-compose (recommended — wires up volumes automatically):
```bash
# one-time: run the full pipeline inside a container to populate data/models
docker compose run --rm trainer

# start the dashboard
docker compose up dashboard

# optional: MLflow UI
docker compose --profile tools up mlflow-ui
```

## 3. Render Deployment

1. Push this repository to GitHub.
2. In Render, create a **new Web Service** from the repo.
3. Environment: **Docker** (Render will use the included `Dockerfile` automatically).
4. Set the service's start command override only if you don't want the default dashboard
   command — otherwise leave it as-is; the Dockerfile's `CMD` already launches Streamlit.
5. Add a **Render Disk** (or run the `trainer` stage as a Render one-off job / build step) so
   `data/`, `models/`, and `mlruns/` persist across deploys — without persistent storage the
   dashboard will show the "no trained model found" notice until the pipeline is re-run.
6. Expose port `8501` (Render detects this from the Dockerfile `EXPOSE` directive).

## 4. Streamlit Community Cloud Deployment

1. Push this repository to GitHub (public or connected private repo).
2. In Streamlit Cloud, create a new app pointing at `dashboard/app.py` on your default branch.
3. Streamlit Cloud installs from `requirements.txt` automatically.
4. Streamlit Cloud's filesystem is ephemeral per deploy — either:
   - commit a pre-generated `data/processed/`, `models/`, and `reports/` (smallest change,
     fine for a demo/portfolio deployment), or
   - add a startup hook that runs `python main.py --stage all` before Streamlit serves
     traffic (better for a "live" deployment that should reflect fresh data).

## Environment Variables

None are required for the default synthetic-data setup. If you connect a real, credentialed
data source in `src/ingestion/data_ingestion.py`, store its credentials as environment
variables (e.g. `DATA_API_KEY`) — never commit them, and add the variable name to a `.env.example`
file for collaborators.
