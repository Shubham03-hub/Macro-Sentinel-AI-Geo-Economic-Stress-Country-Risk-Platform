"""
Business-rule data validation: value ranges, duplicate keys, missingness
thresholds, and target-variable sanity checks. Runs after schema_validation
and produces a structured report consumed by the ingestion->validation stage
of main.py (and logged to reports/ for audit purposes).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.utils.helper import load_config, resolve_path, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataValidationReport:
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    duplicate_keys: int = 0
    missing_ratio: Dict[str, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"type": "issue", "message": m} for m in self.issues]
        rows += [{"type": "warning", "message": m} for m in self.warnings]
        return pd.DataFrame(rows)


class DataValidator:
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.dv_cfg = self.config["data_validation"]

    def validate(self, df: pd.DataFrame) -> DataValidationReport:
        issues: List[str] = []
        warnings: List[str] = []

        # --- duplicate keys -------------------------------------------------
        dup_count = df.duplicated(subset=["iso3", "year"]).sum()
        if dup_count > 0:
            issues.append(f"{dup_count} duplicate (iso3, year) rows found")

        # --- year range -------------------------------------------------
        min_year, max_year = self.dv_cfg["min_year"], self.dv_cfg["max_year"]
        out_of_range_years = df[(df["year"] < min_year) | (df["year"] > max_year)]
        if len(out_of_range_years) > 0:
            issues.append(f"{len(out_of_range_years)} rows have year outside [{min_year}, {max_year}]")

        # --- missingness per column -------------------------------------------------
        missing_ratio = {}
        max_missing = self.dv_cfg["max_missing_ratio_per_column"]
        for col in self.dv_cfg["required_indicator_columns"]:
            if col not in df.columns:
                continue
            ratio = df[col].isna().mean()
            missing_ratio[col] = round(float(ratio), 4)
            if ratio > max_missing:
                issues.append(f"Column '{col}' has {ratio:.1%} missing values (limit {max_missing:.0%})")
            elif ratio > 0:
                warnings.append(f"Column '{col}' has {ratio:.1%} missing values")

        # --- target variable sanity -------------------------------------------------
        target_col = self.dv_cfg["target_column"]
        tmin, tmax = self.dv_cfg["target_min"], self.dv_cfg["target_max"]
        if target_col in df.columns:
            target_missing = df[target_col].isna().sum()
            if target_missing > 0:
                warnings.append(f"{target_missing} rows missing target '{target_col}' (will be dropped pre-training)")

            out_of_bounds = df[(df[target_col] < tmin) | (df[target_col] > tmax)]
            if len(out_of_bounds.dropna(subset=[target_col])) > 0:
                issues.append(f"{len(out_of_bounds)} target values outside [{tmin}, {tmax}]")
        else:
            issues.append(f"Target column '{target_col}' not present")

        # --- referential integrity: every country has a metadata match -------------------------------------------------
        if "region" in df.columns:
            unmatched = df["region"].isna().sum()
            if unmatched > 0:
                warnings.append(f"{unmatched} rows have no matching country_metadata record")

        passed = len(issues) == 0

        report = DataValidationReport(
            passed=passed,
            issues=issues,
            warnings=warnings,
            row_count=len(df),
            duplicate_keys=int(dup_count),
            missing_ratio=missing_ratio,
        )

        if passed:
            logger.info(f"Data validation PASSED — {len(df)} rows, {len(warnings)} warnings")
        else:
            logger.error(f"Data validation FAILED — {len(issues)} issues: {issues}")

        return report

    def save_report(self, report: DataValidationReport) -> None:
        reports_dir = resolve_path(self.config["paths"]["reports_dir"])
        ensure_dir(reports_dir)
        out_path = reports_dir / "data_validation_report.csv"
        report.to_frame().to_csv(out_path, index=False)
        logger.info(f"Validation report saved to {out_path}")


if __name__ == "__main__":
    cfg = load_config()
    master_path = resolve_path(cfg["paths"]["interim_dir"]) / "master_dataset.csv"
    data = pd.read_csv(master_path)

    validator = DataValidator(cfg)
    result = validator.validate(data)
    validator.save_report(result)

    print(f"Passed: {result.passed}")
    print(f"Issues: {result.issues}")
    print(f"Warnings: {result.warnings[:5]}{'...' if len(result.warnings) > 5 else ''}")

    if not result.passed:
        raise SystemExit(1)
