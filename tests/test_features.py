"""Unit tests for src.features.feature_engineering."""

import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    def test_lag_features_shift_correctly_within_country(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.add_lag_features(sample_master_df)

        usa = result[result["iso3"] == "USA"].sort_values("year").reset_index(drop=True)
        # lag1 at row i should equal the raw value at row i-1
        assert np.isclose(usa.loc[1, "gdp_growth_pct_lag1"], usa.loc[0, "gdp_growth_pct"])
        assert pd.isna(usa.loc[0, "gdp_growth_pct_lag1"])  # no prior year for the first observation

    def test_lag_features_do_not_leak_across_countries(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.add_lag_features(sample_master_df.sort_values(["iso3", "year"]))

        # first observation of every country must have NaN lag features
        first_rows = result.sort_values("year").groupby("iso3").head(1)
        assert first_rows["gdp_growth_pct_lag1"].isna().all()

    def test_rolling_features_use_only_past_values(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.add_rolling_features(sample_master_df)
        usa = result[result["iso3"] == "USA"].sort_values("year").reset_index(drop=True)
        # rolling mean at the first row must be NaN (shift(1) with no history)
        assert pd.isna(usa.loc[0, "gdp_growth_pct_rollmean3"])

    def test_heuristic_risk_score_is_present_and_numeric(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.add_heuristic_risk_feature(sample_master_df)
        assert "heuristic_risk_score" in result.columns
        assert pd.api.types.is_numeric_dtype(result["heuristic_risk_score"])

    def test_run_produces_more_columns_than_input(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.run(sample_master_df)
        assert result.shape[1] > sample_master_df.shape[1]
        assert result.shape[0] == sample_master_df.shape[0]  # no row loss

    def test_get_feature_columns_excludes_identifiers(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.run(sample_master_df)
        feature_cols = fe.get_feature_columns(result)
        for excluded in ("iso3", "year", "economic_stress_score", "country", "region"):
            assert excluded not in feature_cols

    def test_interaction_features_created(self, config, sample_master_df):
        fe = FeatureEngineer(config)
        result = fe.add_interaction_features(sample_master_df)
        assert "misery_index" in result.columns
        expected = sample_master_df["inflation_pct"] + sample_master_df["unemployment_pct"]
        assert np.allclose(result["misery_index"], expected)
