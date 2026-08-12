"""
Schema-level validation: verifies the merged master dataset has the columns
the rest of the pipeline assumes, in usable dtypes, before any business-rule
checks run. Fails loud and early rather than letting a silent column rename
upstream surface as a confusing model error three stages later.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaValidationResult:
    passed: bool
    missing_columns: List[str] = field(default_factory=list)
    bad_dtype_columns: Dict[str, str] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError(
                f"Schema validation failed. Missing columns: {self.missing_columns}. "
                f"Dtype issues: {self.bad_dtype_columns}"
            )


class SchemaValidator:
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.required_columns = self.config["data_validation"]["required_indicator_columns"]
        self.target_column = self.config["data_validation"]["target_column"]

    def validate(self, df: pd.DataFrame) -> SchemaValidationResult:
        missing = [c for c in self.required_columns + [self.target_column] if c not in df.columns]

        bad_dtypes = {}
        numeric_expected = [
            c for c in self.required_columns
            if c not in ("iso3", "country", "year")
        ]
        for col in numeric_expected:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                bad_dtypes[col] = str(df[col].dtype)

        if "year" in df.columns and not pd.api.types.is_numeric_dtype(df["year"]):
            bad_dtypes["year"] = str(df["year"].dtype)

        passed = len(missing) == 0 and len(bad_dtypes) == 0

        if passed:
            logger.info("Schema validation PASSED")
        else:
            logger.error(f"Schema validation FAILED — missing: {missing}, dtype issues: {bad_dtypes}")

        return SchemaValidationResult(passed=passed, missing_columns=missing, bad_dtype_columns=bad_dtypes)


if __name__ == "__main__":
    from src.utils.helper import resolve_path

    cfg = load_config()
    master_path = resolve_path(cfg["paths"]["interim_dir"]) / "master_dataset.csv"
    data = pd.read_csv(master_path)
    result = SchemaValidator(cfg).validate(data)
    print(result)
    result.raise_if_failed()
