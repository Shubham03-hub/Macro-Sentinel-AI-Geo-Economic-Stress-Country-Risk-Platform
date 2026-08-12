"""Shared pytest fixtures: a small synthetic panel dataset used across
ingestion, validation, feature, and training tests, so tests don't depend
on the real data/raw/ CSVs being present."""

import numpy as np
import pandas as pd
import pytest

from src.utils.helper import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def sample_master_df():
    rng = np.random.default_rng(7)
    countries = [("USA", "United States"), ("IND", "India"), ("NGA", "Nigeria")]
    years = list(range(2015, 2023))

    rows = []
    for iso3, name in countries:
        for year in years:
            rows.append({
                "iso3": iso3, "country": name, "year": year,
                "region": "Test Region", "income_group": "Test Income", "archetype": "emerging",
                "gdp_growth_pct": rng.normal(3, 1),
                "inflation_pct": rng.normal(5, 2),
                "unemployment_pct": rng.normal(7, 1.5),
                "debt_to_gdp_pct": rng.normal(55, 10),
                "current_account_pct_gdp": rng.normal(-1, 1.5),
                "fx_volatility_index": abs(rng.normal(8, 3)),
                "fx_reserves_months_imports": abs(rng.normal(5, 1.5)),
                "political_stability_index": rng.normal(0, 0.5),
                "government_effectiveness_index": rng.normal(0, 0.5),
                "trade_openness_pct_gdp": rng.normal(45, 10),
                "fdi_net_inflows_pct_gdp": rng.normal(2, 1),
                "gdp_per_capita_usd": abs(rng.normal(8000, 2000)),
                "economic_stress_score": float(np.clip(rng.normal(45, 15), 1, 99)),
            })
    return pd.DataFrame(rows)
