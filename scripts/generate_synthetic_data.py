"""
Synthetic data generator for the Geo-Economic Stress Prediction platform.

No production feed was available at build time, so this script fabricates a
World-Bank-style panel dataset with realistic ranges, correlations, and
country archetypes (advanced / emerging / frontier / commodity-dependent).
Swap this out for a real ingestion source once one is available — the
downstream pipeline only cares about the four CSV schemas below.

Run:
    python scripts/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

YEARS = list(range(2005, 2024))

# (country, iso3, region, income_group, archetype)
COUNTRIES = [
    ("United States", "USA", "North America", "High income", "advanced"),
    ("Germany", "DEU", "Europe & Central Asia", "High income", "advanced"),
    ("Japan", "JPN", "East Asia & Pacific", "High income", "advanced"),
    ("United Kingdom", "GBR", "Europe & Central Asia", "High income", "advanced"),
    ("France", "FRA", "Europe & Central Asia", "High income", "advanced"),
    ("Canada", "CAN", "North America", "High income", "advanced"),
    ("South Korea", "KOR", "East Asia & Pacific", "High income", "advanced"),
    ("Australia", "AUS", "East Asia & Pacific", "High income", "advanced"),
    ("Netherlands", "NLD", "Europe & Central Asia", "High income", "advanced"),
    ("Switzerland", "CHE", "Europe & Central Asia", "High income", "advanced"),
    ("China", "CHN", "East Asia & Pacific", "Upper middle income", "emerging"),
    ("India", "IND", "South Asia", "Lower middle income", "emerging"),
    ("Brazil", "BRA", "Latin America & Caribbean", "Upper middle income", "emerging"),
    ("Mexico", "MEX", "Latin America & Caribbean", "Upper middle income", "emerging"),
    ("Indonesia", "IDN", "East Asia & Pacific", "Upper middle income", "emerging"),
    ("Turkey", "TUR", "Europe & Central Asia", "Upper middle income", "emerging"),
    ("South Africa", "ZAF", "Sub-Saharan Africa", "Upper middle income", "emerging"),
    ("Poland", "POL", "Europe & Central Asia", "High income", "emerging"),
    ("Thailand", "THA", "East Asia & Pacific", "Upper middle income", "emerging"),
    ("Vietnam", "VNM", "East Asia & Pacific", "Lower middle income", "emerging"),
    ("Nigeria", "NGA", "Sub-Saharan Africa", "Lower middle income", "frontier"),
    ("Egypt", "EGY", "Middle East & North Africa", "Lower middle income", "frontier"),
    ("Pakistan", "PAK", "South Asia", "Lower middle income", "frontier"),
    ("Bangladesh", "BGD", "South Asia", "Lower middle income", "frontier"),
    ("Kenya", "KEN", "Sub-Saharan Africa", "Lower middle income", "frontier"),
    ("Ghana", "GHA", "Sub-Saharan Africa", "Lower middle income", "frontier"),
    ("Sri Lanka", "LKA", "South Asia", "Lower middle income", "frontier"),
    ("Ethiopia", "ETH", "Sub-Saharan Africa", "Low income", "frontier"),
    ("Argentina", "ARG", "Latin America & Caribbean", "Upper middle income", "distressed"),
    ("Venezuela", "VEN", "Latin America & Caribbean", "Upper middle income", "distressed"),
    ("Lebanon", "LBN", "Middle East & North Africa", "Upper middle income", "distressed"),
    ("Zimbabwe", "ZWE", "Sub-Saharan Africa", "Lower middle income", "distressed"),
    ("Saudi Arabia", "SAU", "Middle East & North Africa", "High income", "commodity"),
    ("Russia", "RUS", "Europe & Central Asia", "Upper middle income", "commodity"),
    ("Nigeria2", "NGA2", "Sub-Saharan Africa", "Lower middle income", "commodity"),  # placeholder removed below
    ("United Arab Emirates", "ARE", "Middle East & North Africa", "High income", "commodity"),
    ("Chile", "CHL", "Latin America & Caribbean", "High income", "commodity"),
    ("Norway", "NOR", "Europe & Central Asia", "High income", "commodity"),
    ("Kazakhstan", "KAZ", "Europe & Central Asia", "Upper middle income", "commodity"),
    ("Colombia", "COL", "Latin America & Caribbean", "Upper middle income", "emerging"),
    ("Philippines", "PHL", "East Asia & Pacific", "Lower middle income", "emerging"),
    ("Malaysia", "MYS", "East Asia & Pacific", "Upper middle income", "emerging"),
    ("Ukraine", "UKR", "Europe & Central Asia", "Lower middle income", "distressed"),
    ("Iraq", "IRQ", "Middle East & North Africa", "Upper middle income", "distressed"),
    ("Morocco", "MAR", "Middle East & North Africa", "Lower middle income", "frontier"),
    ("Peru", "PER", "Latin America & Caribbean", "Upper middle income", "emerging"),
    ("Romania", "ROU", "Europe & Central Asia", "High income", "emerging"),
    ("Portugal", "PRT", "Europe & Central Asia", "High income", "advanced"),
    ("Greece", "GRC", "Europe & Central Asia", "High income", "distressed"),
    ("Israel", "ISR", "Middle East & North Africa", "High income", "advanced"),
    ("New Zealand", "NZL", "East Asia & Pacific", "High income", "advanced"),
]
COUNTRIES = [c for c in COUNTRIES if c[1] != "NGA2"]

ARCHETYPE_PARAMS = {
    # base_stress, stress_volatility, gdp_growth_mean, inflation_mean, debt_mean
    "advanced":   dict(base=18, vol=4,  gdp=1.8, infl=2.2,  debt=75, unemp=5.5,  fx_vol=3,  cur_acct=1.0, reserves=8.0),
    "emerging":   dict(base=35, vol=8,  gdp=4.2, infl=5.5,  debt=55, unemp=7.5,  fx_vol=8,  cur_acct=-1.5, reserves=6.0),
    "frontier":   dict(base=48, vol=10, gdp=4.8, infl=9.0,  debt=48, unemp=9.5,  fx_vol=12, cur_acct=-3.5, reserves=4.0),
    "commodity":  dict(base=32, vol=12, gdp=3.0, infl=6.5,  debt=40, unemp=6.5,  fx_vol=14, cur_acct=2.5, reserves=9.0),
    "distressed": dict(base=68, vol=15, gdp=0.5, infl=25.0, debt=95, unemp=13.0, fx_vol=25, cur_acct=-5.0, reserves=2.0),
}

records = []
for name, iso3, region, income, archetype in COUNTRIES:
    p = ARCHETYPE_PARAMS[archetype]
    stress_level = p["base"] + RNG.normal(0, p["vol"])
    for year in YEARS:
        shock = 0.0
        if year in (2008, 2009):
            shock += RNG.uniform(8, 18)   # global financial crisis
        if year == 2020:
            shock += RNG.uniform(10, 22)  # covid shock
        if year == 2022:
            shock += RNG.uniform(4, 12) if archetype in ("commodity", "distressed", "frontier") else RNG.uniform(2, 6)

        stress_level = 0.7 * stress_level + 0.3 * p["base"] + RNG.normal(0, p["vol"] * 0.4) + shock * 0.5
        stress_level = float(np.clip(stress_level, 2, 98))

        gdp_growth = p["gdp"] - 0.12 * (stress_level - p["base"]) + RNG.normal(0, 1.4)
        inflation = max(-2, p["infl"] + 0.18 * (stress_level - p["base"]) + RNG.normal(0, 2.0))
        unemployment = max(2, p["unemp"] + 0.08 * (stress_level - p["base"]) + RNG.normal(0, 1.0))
        debt_to_gdp = max(5, p["debt"] + 0.35 * (stress_level - p["base"]) + RNG.normal(0, 5))
        current_account = p["cur_acct"] - 0.05 * (stress_level - p["base"]) + RNG.normal(0, 1.5)
        fx_volatility = max(0.5, p["fx_vol"] + 0.15 * (stress_level - p["base"]) + RNG.normal(0, 3))
        fx_reserves_months = max(0.2, p["reserves"] - 0.04 * (stress_level - p["base"]) + RNG.normal(0, 1.0))
        political_stability_idx = float(np.clip(1.2 - 0.02 * (stress_level - 20) + RNG.normal(0, 0.3), -2.5, 2.5))
        gov_effectiveness_idx = float(np.clip(1.0 - 0.018 * (stress_level - 20) + RNG.normal(0, 0.3), -2.5, 2.5))
        trade_openness = max(10, 60 - 0.1 * (stress_level - p["base"]) + RNG.normal(0, 15))
        fdi_pct_gdp = max(-3, 2.5 - 0.03 * (stress_level - p["base"]) + RNG.normal(0, 1.5))
        population_millions = max(0.3, RNG.lognormal(2.6, 1.4))
        gdp_per_capita = max(300, 45000 * np.exp(-0.025 * (stress_level)) * RNG.uniform(0.7, 1.3))

        records.append(dict(
            country=name, iso3=iso3, year=year,
            gdp_growth_pct=round(gdp_growth, 3),
            inflation_pct=round(inflation, 3),
            unemployment_pct=round(unemployment, 3),
            debt_to_gdp_pct=round(debt_to_gdp, 3),
            current_account_pct_gdp=round(current_account, 3),
            fx_volatility_index=round(fx_volatility, 3),
            fx_reserves_months_imports=round(fx_reserves_months, 3),
            political_stability_index=round(political_stability_idx, 3),
            government_effectiveness_index=round(gov_effectiveness_idx, 3),
            trade_openness_pct_gdp=round(trade_openness, 3),
            fdi_net_inflows_pct_gdp=round(fdi_pct_gdp, 3),
            gdp_per_capita_usd=round(gdp_per_capita, 2),
        ))

indicators_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 1) country_metadata.csv
# ---------------------------------------------------------------------------
metadata_df = pd.DataFrame(
    [{"iso3": iso3, "country": name, "region": region, "income_group": income,
      "archetype": archetype} for name, iso3, region, income, archetype in COUNTRIES]
).drop_duplicates(subset="iso3")
metadata_df.to_csv(RAW_DIR / "country_metadata.csv", index=False)

# ---------------------------------------------------------------------------
# 2) country_year_indicators.csv
# ---------------------------------------------------------------------------
indicator_cols = [c for c in indicators_df.columns if c not in ("country", "iso3", "year")]
missing_mask = RNG.random(indicators_df[indicator_cols].shape) < 0.02
indicators_df_masked = indicators_df.copy()
for i, col in enumerate(indicator_cols):
    indicators_df_masked.loc[missing_mask[:, i], col] = np.nan

indicators_df_masked.to_csv(RAW_DIR / "country_year_indicators.csv", index=False)

# ---------------------------------------------------------------------------
# 3) economic_stress_score.csv  (this is our modelling target)
# ---------------------------------------------------------------------------
merged_for_target = indicators_df.merge(
    metadata_df[["iso3", "archetype"]], on="iso3", how="left"
)
stress_records = []
for _, row in merged_for_target.iterrows():
    p = ARCHETYPE_PARAMS[row["archetype"]]
    raw_score = (
        0.22 * (row["debt_to_gdp_pct"] - 50) / 10
        + 0.20 * (row["inflation_pct"] - 4) / 3
        + 0.16 * (row["unemployment_pct"] - 7) / 2
        - 0.14 * row["gdp_growth_pct"]
        + 0.12 * (row["fx_volatility_index"] - 8) / 3
        - 0.10 * row["fx_reserves_months_imports"]
        - 0.08 * row["current_account_pct_gdp"]
        - 0.10 * row["political_stability_index"]
        - 0.08 * row["government_effectiveness_index"]
        + p["base"] / 20
    )
    stress_score = float(np.clip(50 + raw_score * 9 + RNG.normal(0, 3), 1, 100))
    stress_records.append(dict(
        iso3=row["iso3"], country=row["country"], year=row["year"],
        economic_stress_score=round(stress_score, 2),
    ))

stress_df = pd.DataFrame(stress_records)
stress_df.to_csv(RAW_DIR / "economic_stress_score.csv", index=False)

# ---------------------------------------------------------------------------
# 4) indicator_dictionary.csv
# ---------------------------------------------------------------------------
dictionary_rows = [
    ("gdp_growth_pct", "Real GDP growth", "% annual", "Macro / Growth",
     "Annual percentage growth of real GDP. Falling growth is an early warning of stress."),
    ("inflation_pct", "Consumer price inflation", "% annual", "Macro / Prices",
     "Year-over-year CPI inflation. Persistently high inflation erodes purchasing power and signals policy strain."),
    ("unemployment_pct", "Unemployment rate", "% of labor force", "Labor Market",
     "Share of the labor force without work. Rising unemployment is a lagging but reliable stress signal."),
    ("debt_to_gdp_pct", "Government debt to GDP", "% of GDP", "Fiscal",
     "Total government debt as a share of GDP. High and rising ratios raise default and rollover risk."),
    ("current_account_pct_gdp", "Current account balance", "% of GDP", "External",
     "Balance of trade, income and transfers. Persistent deficits indicate external financing dependence."),
    ("fx_volatility_index", "FX volatility index", "index (0-100)", "External",
     "Realized volatility of the local currency vs USD. Spikes often precede balance-of-payments stress."),
    ("fx_reserves_months_imports", "FX reserves coverage", "months of imports", "External",
     "Foreign exchange reserves expressed in months of import cover. Low buffers raise crisis vulnerability."),
    ("political_stability_index", "Political stability index", "index (-2.5 to 2.5)", "Governance",
     "World Bank WGI-style measure of perceived likelihood of political instability or violence."),
    ("government_effectiveness_index", "Government effectiveness index", "index (-2.5 to 2.5)", "Governance",
     "World Bank WGI-style measure of quality of public services and policy implementation."),
    ("trade_openness_pct_gdp", "Trade openness", "% of GDP", "External",
     "Sum of exports and imports as a share of GDP. Captures exposure to global trade shocks."),
    ("fdi_net_inflows_pct_gdp", "FDI net inflows", "% of GDP", "External",
     "Net foreign direct investment inflows. A funding cushion for the current account deficit."),
    ("gdp_per_capita_usd", "GDP per capita", "current USD", "Macro / Growth",
     "Output per person, a broad proxy for development level and shock-absorption capacity."),
    ("economic_stress_score", "Economic stress score (target)", "index (0-100)", "Composite / Target",
     "Composite, model-consumed stress score. Higher values indicate greater macro-financial stress."),
]
dictionary_df = pd.DataFrame(dictionary_rows, columns=[
    "indicator_code", "indicator_name", "unit", "category", "description"
])
dictionary_df.to_csv(RAW_DIR / "indicator_dictionary.csv", index=False)

print("Synthetic raw data written to:", RAW_DIR)
for f in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    print(f" - {f.name}: {df.shape[0]} rows x {df.shape[1]} cols")
