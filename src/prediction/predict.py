"""
Prediction pipeline: loads the champion model bundle and scores the most
recent year of engineered features for every country, producing the four
business deliverables the platform promises:

    1. Risk Score               (0-100, model prediction)
    2. Risk Category             (Low/Moderate/Elevated/High/Severe)
    3. Forecasted Stress Level   (naive next-year forecast via trend + model)
    4. Driver Importance Ranking (global, from the champion model)

Run:
    python -m src.prediction.predict
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.helper import load_config, resolve_path, ensure_dir, categorize_risk
from src.utils.logger import get_logger
from src.utils.model_loader import load_champion_bundle

logger = get_logger(__name__)


def _latest_snapshot(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    return df.sort_values(time_col).groupby(entity_col, as_index=False).tail(1)


def _apply_model(bundle: Dict, X: pd.DataFrame) -> np.ndarray:
    if bundle["metadata"]["needs_scaling"]:
        X_input = pd.DataFrame(bundle["scaler"].transform(X), columns=X.columns, index=X.index)
    else:
        X_input = X
    return bundle["model"].predict(X_input)


def generate_predictions(config: Dict = None, feature_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cfg = config or load_config()
    entity_col = cfg["feature_engineering"]["entity_col"]
    time_col = cfg["feature_engineering"]["time_col"]
    target_col = cfg["feature_engineering"]["target_col"]

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    bundle = load_champion_bundle(models_dir)
    feature_columns = bundle["feature_columns"]

    if feature_df is None:
        processed_dir = resolve_path(cfg["paths"]["processed_dir"])
        feature_df = pd.read_csv(processed_dir / cfg["processed_files"]["feature_dataset"])

    latest = _latest_snapshot(feature_df, entity_col, time_col).copy()
    X = latest[feature_columns].fillna(latest[feature_columns].median(numeric_only=True)).fillna(0)

    latest["predicted_stress_score"] = _apply_model(bundle, X).round(2)
    latest["risk_category"] = latest["predicted_stress_score"].apply(lambda s: categorize_risk(s, cfg))

    # naive one-step-ahead forecast: blend current model prediction with the
    # country's own 3y trailing trend, since we don't have next year's macro
    # inputs yet — this is a transparent placeholder for a proper time-series
    # forecaster (see reports/architecture.md, "Forecasting Extension").
    trend_col = "debt_to_gdp_pct_trend3y"
    if trend_col in latest.columns:
        trend_adj = latest[trend_col].fillna(0) * 0.6
    else:
        trend_adj = 0
    latest["forecasted_stress_next_year"] = (latest["predicted_stress_score"] + trend_adj).clip(0, 100).round(2)
    latest["forecasted_risk_category"] = latest["forecasted_stress_next_year"].apply(
        lambda s: categorize_risk(s, cfg)
    )

    if target_col in latest.columns:
        latest["actual_stress_score"] = latest[target_col]
        latest["prediction_error"] = (latest["actual_stress_score"] - latest["predicted_stress_score"]).round(2)

    output_cols = [
        entity_col, "country", time_col, "region", "income_group", "archetype",
        "predicted_stress_score", "risk_category",
        "forecasted_stress_next_year", "forecasted_risk_category",
    ]
    output_cols = [c for c in output_cols if c in latest.columns]
    if "actual_stress_score" in latest.columns:
        output_cols += ["actual_stress_score", "prediction_error"]

    result = latest[output_cols].sort_values("predicted_stress_score", ascending=False).reset_index(drop=True)

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    ensure_dir(processed_dir)
    out_path = processed_dir / cfg["processed_files"]["predictions"]
    result.to_csv(out_path, index=False)
    logger.info(f"Predictions for {len(result)} countries written to {out_path}")

    return result


def predict_for_country(iso3: str, config: Dict = None) -> Dict:
    """Convenience accessor used by the dashboard / API-style callers."""
    cfg = config or load_config()
    predictions = generate_predictions(cfg)
    row = predictions[predictions["iso3"] == iso3]
    if row.empty:
        raise ValueError(f"No prediction available for iso3='{iso3}'")
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    preds = generate_predictions()
    print(preds.head(15).to_string(index=False))
    print(f"\nTotal countries scored: {len(preds)}")
    print("\nRisk category distribution:")
    print(preds["risk_category"].value_counts())
