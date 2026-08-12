"""Unit tests for src.validation.schema_validation and data_validation."""

import numpy as np
import pandas as pd
import pytest

from src.validation.schema_validation import SchemaValidator
from src.validation.data_validation import DataValidator


class TestSchemaValidator:
    def test_passes_on_well_formed_frame(self, config, sample_master_df):
        result = SchemaValidator(config).validate(sample_master_df)
        assert result.passed
        assert result.missing_columns == []

    def test_fails_on_missing_required_column(self, config, sample_master_df):
        df = sample_master_df.drop(columns=["debt_to_gdp_pct"])
        result = SchemaValidator(config).validate(df)
        assert not result.passed
        assert "debt_to_gdp_pct" in result.missing_columns

    def test_fails_on_bad_dtype(self, config, sample_master_df):
        df = sample_master_df.copy()
        df["gdp_growth_pct"] = df["gdp_growth_pct"].astype(str) + "%"
        result = SchemaValidator(config).validate(df)
        assert not result.passed
        assert "gdp_growth_pct" in result.bad_dtype_columns

    def test_raise_if_failed_raises(self, config, sample_master_df):
        df = sample_master_df.drop(columns=["inflation_pct"])
        result = SchemaValidator(config).validate(df)
        with pytest.raises(ValueError):
            result.raise_if_failed()


class TestDataValidator:
    def test_passes_on_clean_data(self, config, sample_master_df):
        report = DataValidator(config).validate(sample_master_df)
        assert report.passed
        assert report.duplicate_keys == 0

    def test_detects_duplicate_keys(self, config, sample_master_df):
        df = pd.concat([sample_master_df, sample_master_df.iloc[[0]]], ignore_index=True)
        report = DataValidator(config).validate(df)
        assert not report.passed
        assert report.duplicate_keys >= 1

    def test_detects_excessive_missingness(self, config, sample_master_df):
        df = sample_master_df.copy()
        df.loc[: int(len(df) * 0.6), "inflation_pct"] = np.nan
        report = DataValidator(config).validate(df)
        assert not report.passed
        assert any("inflation_pct" in issue for issue in report.issues)

    def test_detects_out_of_bounds_target(self, config, sample_master_df):
        df = sample_master_df.copy()
        df.loc[0, "economic_stress_score"] = 250.0
        report = DataValidator(config).validate(df)
        assert not report.passed
