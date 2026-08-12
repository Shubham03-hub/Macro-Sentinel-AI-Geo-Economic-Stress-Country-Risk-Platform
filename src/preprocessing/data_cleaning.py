"""
Data cleaning: deduplicates, imputes missing indicator values using a
country-level-then-global median fallback (so a single missing year for
Germany doesn't get imputed with a frontier-market median), winsorizes
extreme outliers, and drops rows with no usable target.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.dv_cfg = self.config["data_validation"]
        self.numeric_cols = [
            c for c in self.dv_cfg["required_indicator_columns"]
            if c not in ("iso3", "country", "year")
        ]
        self.target_col = self.dv_cfg["target_column"]

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["iso3", "year"], keep="first").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info(f"Dropped {removed} duplicate (iso3, year) rows")
        return df

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            missing_before = df[col].isna().sum()
            if missing_before == 0:
                continue
            # country-level median first
            df[col] = df.groupby("iso3")[col].transform(lambda s: s.fillna(s.median()))
            # fall back to global median for countries with all-NaN history
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"Imputed {missing_before} missing values in '{col}'")
        return df

    def winsorize_outliers(self, df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
        df = df.copy()
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            lo, hi = df[col].quantile(lower_q), df[col].quantile(upper_q)
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            if n_clipped:
                logger.info(f"Winsorized {n_clipped} outlier values in '{col}' to [{lo:.2f}, {hi:.2f}]")
        return df

    def drop_missing_target(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info(f"Dropped {removed} rows with missing target '{self.target_col}'")
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting cleaning on {len(df)} rows")
        df = self.drop_duplicates(df)
        df = self.drop_missing_target(df)
        df = self.impute_missing(df)
        df = self.winsorize_outliers(df)
        logger.info(f"Cleaning complete: {len(df)} rows remain")
        return df


if __name__ == "__main__":
    from src.utils.helper import resolve_path

    cfg = load_config()
    master_path = resolve_path(cfg["paths"]["interim_dir"]) / "master_dataset.csv"
    data = pd.read_csv(master_path)

    cleaner = DataCleaner(cfg)
    cleaned = cleaner.run(data)

    out_path = resolve_path(cfg["paths"]["interim_dir"]) / "cleaned_dataset.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Cleaned dataset written to {out_path} — shape {cleaned.shape}")
