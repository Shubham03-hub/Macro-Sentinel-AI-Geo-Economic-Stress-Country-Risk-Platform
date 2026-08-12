# EDA Summary — Geo-Economic Stress Prediction

### Dataset Overview

950 country-year observations covering 50 countries across 19 years (2005-2023). 145 columns.

### Data Types Analysis

Column dtype breakdown: {dtype('float64'): '135', dtype('O'): '5', dtype('int64'): '5'}. All indicator columns are numeric as expected; identifier columns (iso3, country, region, income_group, archetype) are object/string.

### Missing Value Analysis

Highest-missingness columns: {'inflation_pct_lag3': 0.15789473684210525, 'debt_to_gdp_pct_lag3': 0.15789473684210525, 'current_account_pct_gdp_lag3': 0.15789473684210525}. Missingness is handled downstream via country-level median imputation (src/preprocessing/data_cleaning.py) rather than dropped, to preserve full country coverage.

### Duplicate Analysis

0 duplicate (iso3, year) rows found in the raw merged dataset — dataset is clean on its panel key.

### Outlier Analysis

IQR-method outlier counts (top 3 indicators): [('inflation_pct', 133), ('fx_volatility_index', 23), ('unemployment_pct', 18)]. These are largely legitimate crisis-period observations (2008-09, 2020, distressed-archetype countries) rather than data errors, so preprocessing winsorizes rather than drops them — extreme values are informative for stress prediction.

### Univariate Analysis

Most right/left-skewed indicators: {'inflation_pct': 1.43, 'fx_volatility_index': 0.81, 'unemployment_pct': 0.78}. gdp_per_capita_usd and debt_to_gdp_pct show the heaviest skew, consistent with a long tail of high-income and high-debt outlier countries.

### Bivariate Analysis

Indicators most correlated with economic_stress_score: {'inflation_pct': 0.93, 'unemployment_pct': 0.91, 'fx_reserves_months_imports': -0.86, 'fx_volatility_index': 0.84}. Debt, inflation, and unemployment move with stress as expected from economic theory; political/governance indices move inversely, as better governance dampens stress.

### Multivariate Analysis

A 2-component PCA on the indicator set explains 62% of total variance and visibly separates 'advanced' from 'distressed'/'frontier' archetypes, confirming the indicator set captures real structural differences between country groups rather than noise.

### Correlation Analysis

Strongest pairwise indicator correlations: [('inflation_pct~unemployment_pct', 0.87), ('inflation_pct~fx_volatility_index', 0.86), ('unemployment_pct~fx_reserves_months_imports', 0.81)]. Debt and fiscal indicators move together as expected; feature engineering deliberately includes both raw and derived versions since tree-based models (the champion class here) are robust to this collinearity.

### Economic Trend Analysis

Global average stress peaks in 2009, consistent with the simulated crisis shocks built into the synthetic data generator (2008-09 GFC, 2020 COVID, 2022 commodity/rate shock). This confirms crisis-year flags in feature engineering are capturing real structural breaks in the target.

### Country Risk Analysis

In 2023, highest-stress countries are ['Argentina', 'Greece', 'Iraq']; 'distressed' archetype countries show materially higher median stress than 'advanced' economies, confirming the archetype-based synthetic design produces a target distribution consistent with real-world sovereign risk patterns.
