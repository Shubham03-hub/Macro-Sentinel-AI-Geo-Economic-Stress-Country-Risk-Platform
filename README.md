
# 🌍 Macro Sentinel — AI Geo-Economic Stress & Country Risk Platform

Macro Sentinel is an end-to-end machine learning platform that scores, ranks, and forecasts
macroeconomic stress for countries — replacing slow, manually-maintained analyst watchlists
with a reproducible, model-driven **Economic Stress Score**, a **Risk Category**, a **forward
stress estimate**, and a transparent breakdown of what's driving the number.

Built for the kind of desk that currently tracks country risk via quarterly rating-agency
updates and spreadsheets: investment committees, sovereign bond desks, treasury teams, and
corporates with cross-border exposure.

---

## What it does

- **Ingests** country-year macroeconomic indicators (GDP growth, inflation, unemployment,
  debt-to-GDP, FX volatility, political stability, governance, trade openness, FDI, and more)
- **Validates & cleans** the panel data, handling missingness with country-level median
  imputation and winsorizing legitimate crisis-period outliers rather than dropping them
- **Engineers features**: lags, rolling windows, z-scores vs. each country's own history, and
  interaction terms
- **Trains and compares 4 models** (Linear Regression, Random Forest, XGBoost, LightGBM) with
  cross-validated hyperparameter search, tracked in **MLflow**
- **Promotes a champion model** and evaluates it on a held-out test set
- **Generates predictions**: Economic Stress Score, Risk Category (Low → Severe), forecasted
  stress trend, and a ranked driver-importance breakdown per country
- **Serves it all** through an interactive **Streamlit** dashboard for benchmarking countries
  and drilling into what's driving each score

### Current champion model performance (held-out test set)

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 2.72 | 3.38 | 0.964 | 4.35% |
| Random Forest | 2.35 | 3.21 | 0.968 | 4.20% |
| XGBoost | 2.29 | 2.97 | 0.972 | 3.99% |
| **LightGBM (champion)** | **2.26** | **3.10** | **0.970** | **3.94%** |

Top drivers of the stress score include `gdp_growth_pct`, `fx_volatility_index`,
`debt_to_gdp_pct`, and `political_stability_index` (each evaluated both in level and as a
z-score vs. the country's own history) — see `reports/driver_importance_top20.csv`.

---

## Architecture

```
Data Sources (data/raw/*.csv)
      │
      ▼
Data Ingestion          → src/ingestion/data_ingestion.py
      │
      ▼
Data Validation         → src/validation/schema_validation.py, data_validation.py
      │
      ▼
Data Cleaning           → src/preprocessing/data_cleaning.py
      │
      ▼
Feature Engineering     → src/features/feature_engineering.py
      │
      ▼
Model Training          → src/training/train_model.py, hyperparameter_tuning.py
      │
      ▼
Experiment Tracking     → MLflow (mlruns/mlflow.db)
      │
      ▼
Evaluation               → src/evaluation/evaluate_model.py
      │
      ▼
Prediction Pipeline      → src/prediction/predict.py
      │
      ▼
Dashboard                → dashboard/app.py (Streamlit)
```

Full write-up in [`reports/architecture.md`](reports/architecture.md), business case in
[`reports/business_understanding.md`](reports/business_understanding.md), and EDA findings in
[`reports/eda_summary.md`](reports/eda_summary.md).

---

## Tech stack

- **Data & ML**: pandas, numpy, scikit-learn, XGBoost, LightGBM
- **Experiment tracking**: MLflow
- **Dashboard**: Streamlit, Plotly
- **Config-driven**: every stage reads from `config/config.yaml` — no hardcoded paths or params
- **Testing**: pytest
- **Deployment**: Docker / docker-compose, Render, Streamlit Community Cloud

---

## Quickstart (local)

> Requires Python 3.10+.

```bash
git clone https://github.com/Shubham03-hub/Macro-Sentinel-AI-Geo-Economic-Stress-Country-Risk-Platform.git
cd Macro-Sentinel-AI-Geo-Economic-Stress-Country-Risk-Platform

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# generate the synthetic country-year dataset (skip if you're dropping in real CSVs)
python scripts/generate_synthetic_data.py

# run the full pipeline: preprocess -> features -> train -> evaluate -> predict
python main.py --stage all

# launch the dashboard
streamlit run dashboard/app.py
```

Dashboard: **http://localhost:8501**
MLflow UI (optional): `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` → **http://localhost:5000**

> ⚠️ **`data/raw/`, `data/interim/`, `data/processed/`, `models/`, `mlruns/`, and `logs/` are
> git-ignored on purpose** (raw data and trained artifacts don't belong in version control).
> That means a fresh clone of this repo has none of them — you must run
> `generate_synthetic_data.py` and `main.py --stage all` once before the dashboard has anything
> to show. This is the #1 reason a freshly cloned copy looks broken: it isn't, it just hasn't
> been run yet.

### Individual pipeline stages

```bash
python main.py --stage preprocess
python main.py --stage features
python main.py --stage train
python main.py --stage evaluate
python main.py --stage predict
```

### Run tests

```bash
pytest
```

---

## Docker

```bash
# one-time: populate data/models by running the pipeline inside a container
docker compose run --rm trainer

# start the dashboard
docker compose up dashboard

# optional: MLflow UI
docker compose --profile tools up mlflow-ui
```

Or without compose:
```bash
docker build -t macro-sentinel .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models macro-sentinel
```

---

## Deploying the dashboard publicly (Render / Streamlit Cloud)

Both platforms need `data/processed/`, `models/`, and `mlruns/` to exist since they're not in
the repo. Two options:

1. **Commit a pre-generated snapshot** of `data/processed/`, `models/`, and `reports/` (simplest
   — fine for a portfolio/demo deployment that doesn't need to retrain live), or
2. **Run `python main.py --stage all` as a build/startup step** so the deployment always
   reflects a fresh pipeline run.

Full platform-specific steps (Render disk config, Streamlit Cloud settings) are in
[`reports/deployment_guide.md`](reports/deployment_guide.md).

---

## Project structure

```
macro-sentinel/
├── main.py                    # single CLI entrypoint (--stage preprocess|features|train|evaluate|predict|all)
├── config/config.yaml         # all paths, params, model grids, risk thresholds — single source of truth
├── src/
│   ├── ingestion/              # loads + joins the 4 raw CSVs
│   ├── validation/             # schema + data quality checks
│   ├── preprocessing/          # cleaning, imputation, winsorizing
│   ├── features/                # lags, rolling windows, z-scores, interactions
│   ├── training/                # model training + hyperparameter search
│   ├── evaluation/              # champion model evaluation
│   ├── prediction/              # scoring + risk categorization
│   ├── monitoring/              # pipeline/data monitoring
│   └── utils/                    # config loader, logger, model loader, helpers
├── dashboard/app.py            # Streamlit executive dashboard
├── scripts/generate_synthetic_data.py
├── notebooks/01_eda.py
├── reports/                     # business case, architecture, EDA, deployment guide, metrics
├── tests/                       # pytest suite
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

---

## Dataset

Four CSVs are expected in `data/raw/` (World-Bank-style schema; no live feed was available at
build time, so `scripts/generate_synthetic_data.py` fabricates all four realistically — swap in
a licensed feed later and nothing downstream changes, since every stage reads only the CSV
schema, not the generator):

| File | Contents |
|---|---|
| `country_metadata.csv` | Static country attributes (region, income group, archetype) |
| `country_year_indicators.csv` | Annual macro panel — GDP growth, inflation, unemployment, debt, FX, governance, trade, FDI |
| `economic_stress_score.csv` | Modelling target |
| `indicator_dictionary.csv` | Human-readable indicator definitions (used for docs/dashboard tooltips) |

Current synthetic dataset: 950 country-year observations, 50 countries, 2005–2023.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).