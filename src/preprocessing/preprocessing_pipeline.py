"""
Orchestrates ingestion -> schema validation -> data validation -> cleaning
into a single callable, so main.py and the dashboard don't each re-implement
the wiring. Returns the cleaned, merged country-year dataset ready for
feature engineering.
"""

from typing import Dict

import pandas as pd

from src.ingestion.data_ingestion import DataIngestion
from src.validation.schema_validation import SchemaValidator
from src.validation.data_validation import DataValidator
from src.preprocessing.data_cleaning import DataCleaner
from src.utils.helper import load_config, resolve_path, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing_pipeline(config: Dict = None, strict: bool = True) -> pd.DataFrame:
    cfg = config or load_config()

    logger.info("=== STAGE 1/3: Ingestion ===")
    master_df = DataIngestion(cfg).run()

    logger.info("=== STAGE 2/3: Validation ===")
    schema_result = SchemaValidator(cfg).validate(master_df)
    schema_result.raise_if_failed()

    validator = DataValidator(cfg)
    validation_report = validator.validate(master_df)
    validator.save_report(validation_report)
    if strict and not validation_report.passed:
        raise ValueError(f"Data validation failed: {validation_report.issues}")

    logger.info("=== STAGE 3/3: Cleaning ===")
    cleaned_df = DataCleaner(cfg).run(master_df)

    interim_dir = resolve_path(cfg["paths"]["interim_dir"])
    ensure_dir(interim_dir)
    cleaned_df.to_csv(interim_dir / "cleaned_dataset.csv", index=False)

    logger.info(f"Preprocessing pipeline complete: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} cols")
    return cleaned_df


if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print(df.head())
    print(f"\nFinal shape: {df.shape}")
