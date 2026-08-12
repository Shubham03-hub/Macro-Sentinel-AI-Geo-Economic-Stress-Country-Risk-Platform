"""
Post-training evaluation: generates the metrics table, residual plot,
prediction-vs-actual plot, error distribution, and a feature/driver
importance ranking for the champion model. Everything is written to
reports/ and reports/figures/ so it can be dropped straight into a
stakeholder deck.
"""

from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.helper import load_config, resolve_path, ensure_dir
from src.utils.logger import get_logger
from src.utils.model_loader import load_artifact

logger = get_logger(__name__)


def _load_test_set(cfg: Dict) -> pd.DataFrame:
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    return pd.read_csv(processed_dir / cfg["processed_files"]["test_set"])


def _driver_importance(model, feature_columns) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.zeros(len(feature_columns))

    df = pd.DataFrame({"feature": feature_columns, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
    return df


def evaluate_champion(config: Dict = None) -> Dict:
    cfg = config or load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    ensure_dir(reports_dir)
    ensure_dir(figures_dir)

    metadata = load_artifact(models_dir / "champion_metadata.joblib")
    model = load_artifact(models_dir / "champion_model.joblib")
    scaler = load_artifact(models_dir / "feature_scaler.joblib")
    feature_columns = load_artifact(models_dir / "feature_columns.joblib")
    target_col = cfg["feature_engineering"]["target_col"]

    test_df = _load_test_set(cfg)
    X_test = test_df[feature_columns].fillna(test_df[feature_columns].median(numeric_only=True)).fillna(0)
    y_test = test_df[target_col].values

    if metadata["needs_scaling"]:
        X_input = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_input = X_test

    y_pred = model.predict(X_input)
    residuals = y_test - y_pred

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
        "mape": float(np.mean(np.abs(residuals / np.clip(np.abs(y_test), 1e-6, None))) * 100),
    }
    logger.info(f"Champion model ({metadata['champion_name']}) evaluation: {metrics}")

    # --- Prediction vs Actual --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Stress Score")
    ax.set_ylabel("Predicted Stress Score")
    ax.set_title(f"Prediction vs Actual — {metadata['champion_name']}")
    fig.tight_layout()
    fig.savefig(figures_dir / "prediction_vs_actual.png", dpi=150)
    plt.close(fig)

    # --- Residual plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Stress Score")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    fig.savefig(figures_dir / "residual_plot.png", dpi=150)
    plt.close(fig)

    # --- Error distribution --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "error_distribution.png", dpi=150)
    plt.close(fig)

    # --- Driver importance --------------------------------------------------
    importance_df = _driver_importance(model, feature_columns)
    importance_df.head(20).to_csv(reports_dir / "driver_importance_top20.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    top15 = importance_df.head(15).iloc[::-1]
    ax.barh(top15["feature"], top15["importance_pct"])
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top 15 Stress Drivers — {metadata['champion_name']}")
    fig.tight_layout()
    fig.savefig(figures_dir / "driver_importance.png", dpi=150)
    plt.close(fig)

    # --- Metrics table / business summary --------------------------------------------------
    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "model", metadata["champion_name"])
    metrics_df.to_csv(reports_dir / "champion_evaluation_metrics.csv", index=False)

    logger.info(f"Evaluation artifacts written to {reports_dir} and {figures_dir}")
    return {"metrics": metrics, "driver_importance": importance_df, "champion_name": metadata["champion_name"]}


if __name__ == "__main__":
    result = evaluate_champion()
    print(f"Champion: {result['champion_name']}")
    print(f"Metrics: {result['metrics']}")
    print("\nTop 10 drivers:")
    print(result["driver_importance"].head(10).to_string(index=False))
