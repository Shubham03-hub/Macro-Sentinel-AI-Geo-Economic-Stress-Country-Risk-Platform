"""
Data ingestion: reads the four raw CSVs (country_metadata, country_year_
indicators, economic_stress_score, indicator_dictionary), merges them into a
single country-year master table, and writes it to data/interim/. This is
the only module allowed to touch data/raw/ directly — everything downstream
reads from data/interim/master_dataset.csv.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.helper import load_config, resolve_path, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.raw_dir = resolve_path(self.config["paths"]["raw_dir"])
        self.interim_dir = resolve_path(self.config["paths"]["interim_dir"])
        ensure_dir(self.interim_dir)

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected raw file not found: {path}. "
                f"Run scripts/generate_synthetic_data.py or drop real extracts here."
            )
        df = pd.read_csv(path)
        logger.info(f"Loaded {filename}: {df.shape[0]} rows x {df.shape[1]} cols")
        return df

    def load_raw_tables(self) -> Dict[str, pd.DataFrame]:
        files = self.config["raw_files"]
        return {
            "metadata": self._read_csv(files["country_metadata"]),
            "indicators": self._read_csv(files["country_year_indicators"]),
            "stress_score": self._read_csv(files["economic_stress_score"]),
            "dictionary": self._read_csv(files["indicator_dictionary"]),
        }

    def merge_tables(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        indicators = tables["indicators"]
        metadata = tables["metadata"]
        stress = tables["stress_score"]

        merged = indicators.merge(
            metadata, on="iso3", how="left", suffixes=("", "_meta")
        )
        if "country_meta" in merged.columns:
            merged.drop(columns=["country_meta"], inplace=True)

        merged = merged.merge(
            stress[["iso3", "year", "economic_stress_score"]],
            on=["iso3", "year"],
            how="left",
        )

        unmatched_meta = merged["region"].isna().sum() if "region" in merged.columns else 0
        unmatched_target = merged["economic_stress_score"].isna().sum()
        if unmatched_meta:
            logger.warning(f"{unmatched_meta} rows have no matching country_metadata entry")
        if unmatched_target:
            logger.warning(f"{unmatched_target} rows have no matching economic_stress_score entry")

        logger.info(f"Merged master dataset: {merged.shape[0]} rows x {merged.shape[1]} cols")
        return merged

    def run(self) -> pd.DataFrame:
        logger.info("Starting data ingestion")
        tables = self.load_raw_tables()
        master = self.merge_tables(tables)

        out_path = self.interim_dir / "master_dataset.csv"
        master.to_csv(out_path, index=False)
        logger.info(f"Master dataset written to {out_path}")
        return master


if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.run()
    print(df.head())
    print(f"\nShape: {df.shape}")
