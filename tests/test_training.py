"""Unit tests for src.training modules. Uses tiny model configs so the suite
stays fast — hyperparameter search grids are overridden per-test."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features.feature_engineering import FeatureEngineer
from src.training.train_model import _impute_features, _regression_metrics
from src.training.hyperparameter_tuning import tune_random_forest


class TestTrainingHelpers:
    def test_impute_features_removes_all_nans(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, 5.0]})
        result = _impute_features(X)
        assert result.isna().sum().sum() == 0

    def test_regression_metrics_keys_and_values(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 37.0])
        metrics = _regression_metrics(y_true, y_pred)
        for key in ("mae", "mse", "rmse", "r2", "mape"):
            assert key in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE always holds

    def test_regression_metrics_zero_for_perfect_prediction(self):
        y = np.array([5.0, 10.0, 15.0])
        metrics = _regression_metrics(y, y.copy())
        assert np.isclose(metrics["mae"], 0)
        assert np.isclose(metrics["rmse"], 0)
        assert np.isclose(metrics["r2"], 1.0)


class TestHyperparameterTuning:
    def test_tune_random_forest_returns_fitted_estimator(self, config, sample_master_df):
        cfg = dict(config)
        cfg["training"] = dict(config["training"])
        cfg["training"]["n_splits_cv"] = 2
        cfg["training"]["hyperparameter_search"] = {"method": "random", "n_iter_random": 2}
        cfg["training"]["random_forest_grid"] = {"n_estimators": [10, 20], "max_depth": [3, 5], "min_samples_leaf": [1]}

        fe = FeatureEngineer(config)
        featured = fe.run(sample_master_df)
        feature_cols = fe.get_feature_columns(featured)
        X = featured[feature_cols].fillna(0)
        y = featured[config["feature_engineering"]["target_col"]].fillna(featured[config["feature_engineering"]["target_col"]].mean())

        model, params = tune_random_forest(X, y, cfg)
        assert hasattr(model, "predict")
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert "n_estimators" in params


class TestLinearBaseline:
    def test_linear_regression_fits_and_predicts(self, sample_master_df):
        X = sample_master_df[["gdp_growth_pct", "inflation_pct"]].fillna(0)
        y = sample_master_df["economic_stress_score"]
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
