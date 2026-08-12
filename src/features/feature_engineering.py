"""
Feature engineering for the geo-economic stress model. Everything here is
panel-aware (grouped by iso3, sorted by year) so lag/rolling features never
leak across countries. Feature families:

  - date/year features        : cyclical + crisis-period flags
  - lag features               : t-1, t-2, t-3 of each base indicator
  - rolling mean/std features   : 3y and 5y rolling stats per indicator
  - country-level aggregations : each country's own historical mean/std
  - growth rate features       : year-over-year % change
  - trend features              : slope of a short trailing window
  - volatility features         : coefficient of variation over trailing window
  - interaction features        : economically-motivated cross terms
  - risk score features         : a simple, transparent composite pre-model score

The composite "heuristic risk feature" is intentionally included as a model
input, not a replacement for the ML model — it gives the model a strong
prior and gives humans a sanity-checkable baseline to compare predictions
against.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.fe_cfg = self.config["feature_engineering"]
        self.entity_col = self.fe_cfg["entity_col"]
        self.time_col = self.fe_cfg["time_col"]
        self.target_col = self.fe_cfg["target_col"]
        self.base_features = self.fe_cfg["base_numeric_features"]
        self.lag_periods = self.fe_cfg["lag_periods"]
        self.rolling_windows = self.fe_cfg["rolling_windows"]

    # ------------------------------------------------------------------ #
    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["years_since_2000"] = df[self.time_col] - 2000
        df["is_crisis_year_2008_09"] = df[self.time_col].isin([2008, 2009]).astype(int)
        df["is_covid_year"] = (df[self.time_col] == 2020).astype(int)
        df["decade"] = (df[self.time_col] // 10) * 10
        logger.info("Added date/year features")
        return df

    # ------------------------------------------------------------------ #
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([self.entity_col, self.time_col]).copy()
        grouped = df.groupby(self.entity_col)
        for col in self.base_features:
            for lag in self.lag_periods:
                df[f"{col}_lag{lag}"] = grouped[col].shift(lag)
        logger.info(f"Added lag features for {len(self.base_features)} indicators x {self.lag_periods} periods")
        return df

    # ------------------------------------------------------------------ #
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([self.entity_col, self.time_col]).copy()
        grouped = df.groupby(self.entity_col)
        for col in self.base_features:
            for window in self.rolling_windows:
                df[f"{col}_rollmean{window}"] = grouped[col].transform(
                    lambda s: s.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f"{col}_rollstd{window}"] = grouped[col].transform(
                    lambda s: s.shift(1).rolling(window, min_periods=2).std()
                )
        logger.info(f"Added rolling mean/std features for windows {self.rolling_windows}")
        return df

    # ------------------------------------------------------------------ #
    def add_country_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.base_features:
            country_mean = df.groupby(self.entity_col)[col].transform("mean")
            country_std = df.groupby(self.entity_col)[col].transform("std").replace(0, np.nan)
            df[f"{col}_country_mean"] = country_mean
            df[f"{col}_zscore_vs_own_history"] = (df[col] - country_mean) / country_std
        logger.info("Added country-level aggregation features")
        return df

    # ------------------------------------------------------------------ #
    def add_growth_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([self.entity_col, self.time_col]).copy()
        grouped = df.groupby(self.entity_col)
        growth_candidates = ["gdp_per_capita_usd", "debt_to_gdp_pct", "fx_reserves_months_imports"]
        for col in growth_candidates:
            if col not in df.columns:
                continue
            prev = grouped[col].shift(1)
            df[f"{col}_yoy_growth_pct"] = np.where(
                (prev.notna()) & (prev != 0), (df[col] - prev) / prev.abs() * 100, np.nan
            )
        logger.info("Added year-over-year growth rate features")
        return df

    # ------------------------------------------------------------------ #
    def add_trend_features(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        df = df.sort_values([self.entity_col, self.time_col]).copy()

        def _rolling_slope(series: pd.Series) -> pd.Series:
            def slope(vals: np.ndarray) -> float:
                if len(vals) < 2 or np.all(np.isnan(vals)):
                    return np.nan
                x = np.arange(len(vals))
                mask = ~np.isnan(vals)
                if mask.sum() < 2:
                    return np.nan
                return float(np.polyfit(x[mask], vals[mask], 1)[0])
            return series.shift(1).rolling(window, min_periods=2).apply(slope, raw=True)

        grouped = df.groupby(self.entity_col)
        for col in ["debt_to_gdp_pct", "inflation_pct", "gdp_growth_pct"]:
            if col not in df.columns:
                continue
            df[f"{col}_trend{window}y"] = grouped[col].transform(_rolling_slope)
        logger.info(f"Added {window}-year trend (slope) features")
        return df

    # ------------------------------------------------------------------ #
    def add_volatility_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = df.sort_values([self.entity_col, self.time_col]).copy()
        grouped = df.groupby(self.entity_col)
        for col in ["fx_volatility_index", "inflation_pct", "gdp_growth_pct"]:
            if col not in df.columns:
                continue
            roll_mean = grouped[col].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
            roll_std = grouped[col].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
            df[f"{col}_cv{window}y"] = roll_std / roll_mean.replace(0, np.nan).abs()
        logger.info(f"Added {window}-year volatility (coefficient of variation) features")
        return df

    # ------------------------------------------------------------------ #
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if {"debt_to_gdp_pct", "gdp_growth_pct"}.issubset(df.columns):
            df["debt_growth_interaction"] = df["debt_to_gdp_pct"] / (df["gdp_growth_pct"].clip(lower=0.1))
        if {"inflation_pct", "unemployment_pct"}.issubset(df.columns):
            df["misery_index"] = df["inflation_pct"] + df["unemployment_pct"]
        if {"current_account_pct_gdp", "fx_reserves_months_imports"}.issubset(df.columns):
            df["external_buffer_score"] = df["fx_reserves_months_imports"] + df["current_account_pct_gdp"].clip(lower=0)
        if {"political_stability_index", "government_effectiveness_index"}.issubset(df.columns):
            df["governance_composite"] = (df["political_stability_index"] + df["government_effectiveness_index"]) / 2
        logger.info("Added economically-motivated interaction features")
        return df

    # ------------------------------------------------------------------ #
    def add_heuristic_risk_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transparent, non-ML composite score — deliberately simple so it can be
        explained to a non-technical stakeholder in one sentence. Included as
        a feature (not a replacement) so the ML models can learn how much to
        trust/adjust it.
        """
        df = df.copy()
        weights = {
            "debt_to_gdp_pct": 0.22,
            "inflation_pct": 0.18,
            "unemployment_pct": 0.15,
            "fx_volatility_index": 0.15,
        }
        z_components = []
        for col, w in weights.items():
            if col not in df.columns:
                continue
            z = (df[col] - df[col].mean()) / df[col].std()
            z_components.append(w * z)
        df["heuristic_risk_score"] = sum(z_components) if z_components else 0.0
        logger.info("Added heuristic (rule-based) composite risk score feature")
        return df

    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering on {len(df)} rows")
        df = self.add_date_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_country_aggregations(df)
        df = self.add_growth_rate_features(df)
        df = self.add_trend_features(df)
        df = self.add_volatility_features(df)
        df = self.add_interaction_features(df)
        df = self.add_heuristic_risk_feature(df)
        logger.info(f"Feature engineering complete: {df.shape[0]} rows x {df.shape[1]} cols")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Every numeric column except identifiers and the raw target."""
        exclude = {self.entity_col, self.time_col, self.target_col, "country", "region", "income_group", "archetype"}
        return [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]


if __name__ == "__main__":
    from src.utils.helper import resolve_path

    cfg = load_config()
    cleaned_path = resolve_path(cfg["paths"]["interim_dir"]) / "cleaned_dataset.csv"
    data = pd.read_csv(cleaned_path)

    engineer = FeatureEngineer(cfg)
    featured = engineer.run(data)

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / cfg["processed_files"]["feature_dataset"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(out_path, index=False)

    feature_cols = engineer.get_feature_columns(featured)
    print(f"Feature dataset written to {out_path} — shape {featured.shape}")
    print(f"Total model-ready feature columns: {len(feature_cols)}")
