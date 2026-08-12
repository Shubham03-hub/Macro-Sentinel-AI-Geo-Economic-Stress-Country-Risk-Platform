"""Unit tests for src.ingestion.data_ingestion."""

import pandas as pd
import pytest

from src.ingestion.data_ingestion import DataIngestion


class TestDataIngestion:
    def test_merge_tables_joins_on_iso3_and_year(self, config):
        ingestion = DataIngestion(config)

        metadata = pd.DataFrame({
            "iso3": ["USA", "IND"], "country": ["United States", "India"],
            "region": ["NA", "SA"], "income_group": ["High", "Low"], "archetype": ["advanced", "emerging"],
        })
        indicators = pd.DataFrame({
            "iso3": ["USA", "USA", "IND"], "country": ["United States", "United States", "India"],
            "year": [2020, 2021, 2020], "gdp_growth_pct": [2.1, 5.6, 4.0],
        })
        stress = pd.DataFrame({
            "iso3": ["USA", "USA", "IND"], "country": ["United States", "United States", "India"],
            "year": [2020, 2021, 2020], "economic_stress_score": [40.0, 35.0, 55.0],
        })
        dictionary = pd.DataFrame({"indicator_code": ["gdp_growth_pct"], "indicator_name": ["GDP growth"]})

        merged = ingestion.merge_tables({
            "metadata": metadata, "indicators": indicators, "stress_score": stress, "dictionary": dictionary,
        })

        assert len(merged) == 3
        assert "economic_stress_score" in merged.columns
        assert "region" in merged.columns
        usa_2020 = merged[(merged["iso3"] == "USA") & (merged["year"] == 2020)].iloc[0]
        assert usa_2020["economic_stress_score"] == 40.0
        assert usa_2020["region"] == "NA"

    def test_merge_tables_flags_unmatched_metadata(self, config, caplog):
        ingestion = DataIngestion(config)
        metadata = pd.DataFrame({
            "iso3": ["USA"], "country": ["United States"], "region": ["NA"],
            "income_group": ["High"], "archetype": ["advanced"],
        })
        indicators = pd.DataFrame({
            "iso3": ["USA", "ZZZ"], "country": ["United States", "Unknown"],
            "year": [2020, 2020], "gdp_growth_pct": [2.1, 1.0],
        })
        stress = pd.DataFrame({
            "iso3": ["USA", "ZZZ"], "country": ["United States", "Unknown"],
            "year": [2020, 2020], "economic_stress_score": [40.0, 60.0],
        })
        dictionary = pd.DataFrame({"indicator_code": ["gdp_growth_pct"], "indicator_name": ["GDP growth"]})

        merged = ingestion.merge_tables({
            "metadata": metadata, "indicators": indicators, "stress_score": stress, "dictionary": dictionary,
        })
        zzz_row = merged[merged["iso3"] == "ZZZ"].iloc[0]
        assert pd.isna(zzz_row["region"])

    def test_read_csv_raises_on_missing_file(self, config, tmp_path):
        ingestion = DataIngestion(config)
        ingestion.raw_dir = tmp_path  # empty dir
        with pytest.raises(FileNotFoundError):
            ingestion._read_csv("does_not_exist.csv")
