# Solution Architecture — Macro Sentinel

## Pipeline Flow

```
Data Sources (data/raw/*.csv)
      |
      v
Data Ingestion            (src/ingestion/data_ingestion.py)
      |
      v
Data Validation            (src/validation/schema_validation.py, data_validation.py)
      |
      v
Data Cleaning               (src/preprocessing/data_cleaning.py)
      |
      v
Feature Engineering         (src/features/feature_engineering.py)
      |
      v
EDA                          (notebooks/, reports/figures/)
      |
      v
Model Training               (src/training/train_model.py, hyperparameter_tuning.py)
      |
      v
Experiment Tracking          (MLflow — mlruns/mlflow.db)
      |
      v
Model Registry / Artifacts   (models/*.joblib)
      |
      v
Evaluation                   (src/evaluation/evaluate_model.py)
      |
      v
Prediction Pipeline          (src/prediction/predict.py)
      |
      v
Dashboard                    (dashboard/app.py — Streamlit)
      |
      v
Deployment                   (Docker / docker-compose / Streamlit Cloud / Render)
```

## Component-by-Component

### Data Sources
Four CSVs assumed present in `data/raw/`: `country_metadata.csv` (static country attributes),
`country_year_indicators.csv` (the annual macro panel), `economic_stress_score.csv` (the
modelling target), `indicator_dictionary.csv` (human-readable definitions, used for
documentation and dashboard tooltips, not consumed by the model itself). No real feed was
available at build time, so `scripts/generate_synthetic_data.py` fabricates all four with a
realistic World-Bank-style schema — swap the script's output for a licensed feed and nothing
downstream changes, since every stage reads only the CSV schema, not the generator.

### Data Ingestion
`DataIngestion` is the only module that touches `data/raw/`. It loads all four CSVs, left-joins
indicators to metadata on `iso3` and to the stress score on `(iso3, year)`, and writes a single
`data/interim/master_dataset.csv`. Centralizing the merge here means every downstream stage
works off one flat table instead of re-deriving the join logic.

### Data Validation
Two layers, deliberately separated:
- **Schema validation** (`schema_validation.py`) — do the required columns exist, are they the
  right dtype? Fails fast before anything expensive runs.
- **Business-rule validation** (`data_validation.py`) — duplicate `(iso3, year)` keys, year
  range sanity, per-column missingness thresholds, target-variable bounds, referential
  integrity against metadata. Produces a structured report saved to
  `reports/data_validation_report.csv` for audit purposes.

### Data Cleaning
`DataCleaner` deduplicates, drops rows with no usable target, imputes missing indicator values
(country-level median first, global median fallback for countries with no history at all — so
one missing Germany observation isn't imputed with a frontier-market median), and winsorizes
extreme outliers at the 1st/99th percentile per column.

### Feature Engineering
`FeatureEngineer` is panel-aware throughout — every lag, rolling, trend, and volatility feature
is grouped by `iso3` and sorted by `year` before computing, with `.shift(1)` applied before any
rolling statistic, so no feature ever sees its own current-year target. Produces ~145 columns
from 12 base indicators: date/crisis-year flags, 1-3 year lags, 3y/5y rolling mean+std,
country-level z-scores, YoY growth rates, trailing trend slopes, volatility (coefficient of
variation), economically-motivated interaction terms (e.g. misery index, external buffer
score), and a transparent rule-based heuristic risk score included as a model input.

### EDA
Covered in `notebooks/01_eda.py` (see Phase 5 deliverable) and `reports/figures/` — dataset
overview, missingness, outliers, univariate/bivariate/multivariate views, correlation
structure, and country risk comparisons.

### Model Training
Four models trained and compared on identical train/test splits: Linear Regression (fast,
fully interpretable baseline), Random Forest, XGBoost, and LightGBM (all three tuned via
K-fold cross-validated randomized/grid search over the grids in `config/config.yaml`). Every
run — parameters, metrics, and the fitted model — is logged to MLflow.

### Experiment Tracking (MLflow)
Tracking URI is a local SQLite database (`mlruns/mlflow.db`) rather than the legacy file store,
which Mlflow has deprecated for new projects. Every training run appears under the
`macro_sentinel` experiment; launch the UI with
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

### Model Registry / Artifacts
The champion (lowest test MAE) model, its fitted `StandardScaler`, the ordered feature column
list, and metadata (which model won, its metrics, and every model's metrics for comparison)
are persisted to `models/*.joblib` via `src/utils/model_loader.py`. All four trained models are
also saved individually so `evaluate_model.py` or an analyst can compare any of them directly.

### Evaluation
`evaluate_champion()` scores the champion model on the held-out test set, writes the metrics
table, a prediction-vs-actual scatter, a residual plot, an error-distribution histogram, and a
global driver-importance ranking (feature importances for tree models, absolute coefficients
for linear) — all to `reports/` and `reports/figures/`.

### Prediction Pipeline
`generate_predictions()` scores the latest available year for every country, mapping the raw
prediction to a business-facing risk category via the configurable bands in
`config/config.yaml` (`risk_scoring.categories`), and produces a naive next-year forecast that
blends the current prediction with the country's own trailing debt-to-GDP trend — an explicit,
transparent placeholder pending a dedicated time-series forecaster (see "Forecasting
Extension" below).

### Dashboard
Streamlit app (`dashboard/app.py`) reading directly from `models/` and
`data/processed/latest_predictions.csv` — no separate API layer for this release. Provides a
world map, KPI cards, a per-country explorer with historical trend + forecast, global driver
importance, risk-category distribution, cross-country indicator comparison, and CSV export.

### Deployment
`Dockerfile` builds a single image containing the full pipeline and dashboard;
`docker-compose.yml` defines a `dashboard` service (long-running, port 8501), a `trainer`
service (`docker compose run --rm trainer` to (re)run the full pipeline), and an `mlflow-ui`
service (port 5000). See `reports/deployment_guide.md` for local, Docker, Render, and Streamlit
Cloud instructions.

## Forecasting Extension (Roadmap, Not Shipped)

The current release predicts the *current* year's stress level from current-year indicators,
and derives a naive next-year forecast from the trailing debt trend. A proper multi-step
forecast (e.g., a per-country ARIMA/Prophet layer feeding the ML model's lag features, or a
direct multi-horizon gradient-boosted quantile model) is the natural next iteration and is
scoped intentionally out of this build to keep the shipped pipeline auditable end-to-end rather
than partially implemented.

## Why This Shape

Each stage writes its output to disk (`data/interim/`, `data/processed/`, `models/`,
`reports/`) rather than passing objects in memory between Python calls. That's a deliberate
trade against a marginally faster in-memory pipeline: it means any stage can be re-run in
isolation (`python -m src.features.feature_engineering`), inspected with `pandas.read_csv`, or
handed to a teammate without needing to replay everything before it — the standard convention
in production ML pipelines, and the reason `main.py --stage <name>` exists rather than a single
monolithic `run()` function.
