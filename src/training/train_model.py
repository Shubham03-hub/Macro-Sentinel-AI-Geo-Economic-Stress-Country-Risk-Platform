"""
Trains and compares Linear Regression, Random Forest, XGBoost, and LightGBM
on the engineered feature set, logs every run to MLflow (params, metrics,
artifacts), registers the best model in the MLflow model registry, and
persists the champion model + scaler + feature list to models/ for the
prediction pipeline and dashboard to consume.

Run:
    python -m src.training.train_model
"""

from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import FeatureEngineer
from src.training.hyperparameter_tuning import tune_lightgbm, tune_random_forest, tune_xgboost
from src.utils.helper import load_config, resolve_path, ensure_dir
from src.utils.logger import get_logger
from src.utils.model_loader import save_artifact

logger = get_logger(__name__)


def _prep_dataset(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    feature_path = processed_dir / config["processed_files"]["feature_dataset"]

    if not feature_path.exists():
        interim_path = resolve_path(config["paths"]["interim_dir"]) / "cleaned_dataset.csv"
        cleaned = pd.read_csv(interim_path)
        engineer = FeatureEngineer(config)
        df = engineer.run(cleaned)
        ensure_dir(processed_dir)
        df.to_csv(feature_path, index=False)
    else:
        df = pd.read_csv(feature_path)

    engineer = FeatureEngineer(config)
    feature_cols = engineer.get_feature_columns(df)
    target_col = config["feature_engineering"]["target_col"]

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df, feature_cols, target_col


def _impute_features(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute remaining NaNs (early-history lag/rolling features)."""
    return X.fillna(X.median(numeric_only=True)).fillna(0)


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


def run_training(config: Dict = None) -> Dict:
    cfg = config or load_config()
    seed = cfg["project"]["random_seed"]

    logger.info("Preparing dataset for training")
    df, feature_cols, target_col = _prep_dataset(cfg)

    X = _impute_features(df[feature_cols])
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["training"]["test_size"], random_state=seed
    )
    logger.info(f"Train/test split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, {X_train.shape[1]} features")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # save train/test sets for the evaluation phase / audit trail
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    ensure_dir(processed_dir)
    train_out = X_train.copy(); train_out[target_col] = y_train
    test_out = X_test.copy(); test_out[target_col] = y_test
    train_out.to_csv(processed_dir / cfg["processed_files"]["train_set"], index=False)
    test_out.to_csv(processed_dir / cfg["processed_files"]["test_set"], index=False)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    cv = KFold(n_splits=cfg["training"]["n_splits_cv"], shuffle=True, random_state=seed)
    results = {}
    fitted_models = {}

    # ---------------------------------------------------------------- #
    # 1. Linear Regression — fast, fully interpretable baseline.
    #    Sets the floor every other model must beat.
    # ---------------------------------------------------------------- #
    with mlflow.start_run(run_name="linear_regression"):
        model = LinearRegression()
        cv_mae = -cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        metrics = _regression_metrics(y_test, preds)
        metrics["cv_mae"] = cv_mae

        mlflow.log_params({"model_type": "linear_regression"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        results["linear_regression"] = metrics
        fitted_models["linear_regression"] = (model, True)  # True = needs scaled input
        logger.info(f"linear_regression -> {metrics}")

    # ---------------------------------------------------------------- #
    # 2. Random Forest — robust to nonlinearity/outliers, minimal tuning
    #    needed, good baseline for feature importance sanity checks.
    # ---------------------------------------------------------------- #
    with mlflow.start_run(run_name="random_forest"):
        best_model, best_params = tune_random_forest(X_train, y_train, cfg)
        cv_mae = -cross_val_score(best_model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        preds = best_model.predict(X_test)
        metrics = _regression_metrics(y_test, preds)
        metrics["cv_mae"] = cv_mae

        mlflow.log_params({f"rf_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

        results["random_forest"] = metrics
        fitted_models["random_forest"] = (best_model, False)
        logger.info(f"random_forest -> {metrics}")

    # ---------------------------------------------------------------- #
    # 3. XGBoost — typically the strongest performer on tabular
    #    macro/financial panel data; handles missing values natively.
    # ---------------------------------------------------------------- #
    with mlflow.start_run(run_name="xgboost"):
        best_model, best_params = tune_xgboost(X_train, y_train, cfg)
        cv_mae = -cross_val_score(best_model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        preds = best_model.predict(X_test)
        metrics = _regression_metrics(y_test, preds)
        metrics["cv_mae"] = cv_mae

        mlflow.log_params({f"xgb_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(best_model, "model")

        results["xgboost"] = metrics
        fitted_models["xgboost"] = (best_model, False)
        logger.info(f"xgboost -> {metrics}")

    # ---------------------------------------------------------------- #
    # 4. LightGBM — comparable accuracy to XGBoost, much faster training,
    #    useful when this pipeline needs to retrain frequently.
    # ---------------------------------------------------------------- #
    with mlflow.start_run(run_name="lightgbm"):
        best_model, best_params = tune_lightgbm(X_train, y_train, cfg)
        cv_mae = -cross_val_score(best_model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        preds = best_model.predict(X_test)
        metrics = _regression_metrics(y_test, preds)
        metrics["cv_mae"] = cv_mae

        mlflow.log_params({f"lgbm_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(best_model, "model")

        results["lightgbm"] = metrics
        fitted_models["lightgbm"] = (best_model, False)
        logger.info(f"lightgbm -> {metrics}")

    # ---------------------------------------------------------------- #
    # Champion selection: lowest test MAE wins.
    # ---------------------------------------------------------------- #
    champion_name = min(results, key=lambda k: results[k]["mae"])
    champion_model, champion_needs_scaling = fitted_models[champion_name]
    logger.info(f"Champion model: {champion_name} (test MAE = {results[champion_name]['mae']:.3f})")

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    ensure_dir(models_dir)

    save_artifact(champion_model, models_dir / "champion_model.joblib")
    save_artifact(scaler, models_dir / "feature_scaler.joblib")
    save_artifact(list(feature_cols), models_dir / "feature_columns.joblib")
    save_artifact(
        {
            "champion_name": champion_name,
            "needs_scaling": champion_needs_scaling,
            "metrics": results[champion_name],
            "all_results": results,
        },
        models_dir / "champion_metadata.joblib",
    )

    # persist every trained model too, so evaluate_model.py can compare all four
    for name, (model, _) in fitted_models.items():
        save_artifact(model, models_dir / f"{name}.joblib")

    results_df = pd.DataFrame(results).T
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    ensure_dir(reports_dir)
    results_df.to_csv(reports_dir / "model_comparison.csv")

    logger.info("Training complete. Model comparison:\n" + results_df.to_string())
    return {"results": results, "champion_name": champion_name}


if __name__ == "__main__":
    run_training()
