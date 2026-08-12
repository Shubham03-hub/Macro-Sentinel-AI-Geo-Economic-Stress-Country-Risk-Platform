# Business Understanding — AI Geo-Economic Stress & Country Risk Platform

## Executive Summary

Investment committees, treasury desks, and government risk offices currently assess country
risk using quarterly analyst reports, credit rating agency updates (which lag real conditions
by months), and manually-maintained watchlists. This platform replaces that latency with a
continuously-updatable, model-driven stress score for every tracked country, backed by a
transparent driver breakdown so the number is defensible in an investment committee meeting,
not just a black box.

## Business Problem

Country-level macroeconomic stress — currency crises, sovereign debt distress, inflation
spirals, political-economic shocks — is expensive to miss and expensive to overreact to.
Firms need an early, quantified signal, not just a post-hoc narrative once a downgrade or
default has already happened.

## Problem Statement

Build a system that ingests country-year macroeconomic indicators and produces, for every
tracked country: a numeric stress score, a risk category, a forward-looking stress estimate,
and a ranked explanation of what is driving it — refreshable as new indicator data lands.

## Current Challenges

- **Lag.** Credit ratings and IMF Article IV consultations are updated infrequently relative
  to how fast currency and debt stress can build.
- **Inconsistency.** Different analysts weight the same indicators differently; scores aren't
  reproducible across teams or over time.
- **Opacity.** Rating agency methodologies are largely proprietary; internal risk teams can't
  audit *why* a score moved.
- **Manual effort.** Building a country risk view today means an analyst manually pulling
  indicators from multiple sources and synthesizing them by hand every cycle.

## Why Organizations Care

- **Asset managers / sovereign bond desks** need early warning before a spread widens or a
  rating action hits, not after.
- **Corporates with cross-border exposure** (suppliers, subsidiaries, receivables) need country
  risk as an input to hedging and working-capital decisions.
- **Development banks / multilaterals** need consistent, explainable risk scoring to justify
  lending terms and conditionality.
- **Insurers (political risk, trade credit)** price coverage directly off country stress
  trajectories.

## Business Impact

A model-driven score that moves weeks ahead of a rating action, and that can be explained in
one sentence ("debt-to-GDP trend and FX volatility are driving this," not "trust the score"),
changes country risk from a quarterly report into a decision-support tool teams can query
on demand.

## Use Cases

1. Sovereign bond desk: rank a coverage universe by stress trajectory before each allocation
   review.
2. Corporate treasury: flag supplier or subsidiary countries crossing into "Elevated" risk
   for hedging or diversification review.
3. Policy / development finance: benchmark a country against peers on the same indicator set
   used for lending decisions.
4. Political risk insurance: feed the forecasted stress trajectory into premium pricing.

## KPIs

| KPI | Target |
|---|---|
| Prediction error (MAE) on held-out country-years | < 5 points on a 0–100 scale |
| Directional accuracy of year-over-year forecast | > 70% |
| Time to refresh scores after new indicator data lands | < 1 hour (automated pipeline run) |
| Analyst time spent building a country risk view | Reduced from ~1 day to < 15 minutes review |

## Success Metrics

- Model MAE and R² on out-of-sample country-years, tracked per retraining cycle in MLflow.
- Stability of driver rankings across retrains (a score that moves for defensible reasons,
  not noise).
- Adoption: number of internal teams pulling from `latest_predictions.csv` / the dashboard
  in a given cycle.

## Assumptions

- Indicator data (macro, fiscal, external, governance) is available at annual country-year
  granularity and can be refreshed on a regular cadence.
- A composite 0–100 stress score is an acceptable abstraction for downstream consumers; teams
  needing raw indicator detail can still drill into the underlying data.
- Historical relationships between indicators and stress outcomes are informative for the
  near-term forecast horizon (1 year); this assumption should be revisited for longer horizons.

## Constraints

- Annual data granularity limits how early a fast-moving currency crisis can be flagged versus
  a system built on monthly/weekly indicators — this is a first-generation platform, not a
  replacement for real-time market signals.
- Model quality depends on indicator completeness; countries with sparse reporting histories
  will have wider uncertainty even after imputation.
- The current release ships with a synthetic dataset (see `scripts/generate_synthetic_data.py`)
  standing in for a licensed indicator feed — production deployment requires connecting a real
  data source (see "Data Source Migration" in `reports/architecture.md`).

## Risks

- **Model risk:** over-reliance on a single score without the driver breakdown could mask
  regime changes the model hasn't seen before (structural breaks, novel shocks).
- **Data risk:** stale or revised source indicators (many macro series are revised months
  after first release) can cause the score to move on a data correction, not a real change.
- **Model drift:** relationships between indicators and stress can shift after major global
  shocks; retraining cadence and monitoring need to be enforced operationally, not left ad hoc.

## Expected ROI

Replacing a multi-day, multi-analyst quarterly review cycle with an automated pipeline that
refreshes in under an hour reduces the fully-loaded analyst cost of maintaining a country risk
view meaningfully, while extending consistent coverage to countries too small to justify a
dedicated analyst under the manual process.

## Cost-Benefit Analysis

| Cost | Benefit |
|---|---|
| Data licensing (if replacing synthetic data with a paid indicator feed) | Consistent, always-on coverage across the full country universe |
| Initial build + MLOps setup (this repository) | Reusable pipeline — new countries/indicators are additive, not a rebuild |
| Ongoing retraining/monitoring compute | Early-warning lead time versus quarterly manual review |
| Change management (teams adopting a model score alongside/instead of analyst judgment) | Reproducible, auditable risk scores instead of tribal-knowledge judgment calls |
