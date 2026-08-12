# Contributing

Thanks for considering a contribution to Macro Sentinel.

## Getting set up

```bash
git clone <repo-url>
cd macro-sentinel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_synthetic_data.py   # or drop real extracts into data/raw/
python main.py --stage all
pytest tests/ -v
```

## Project conventions

- **Config, not constants.** Any tunable value (paths, thresholds, hyperparameter grids, risk-category cutoffs) belongs in `config/config.yaml`, not hardcoded in a module.
- **One responsibility per module.** `src/ingestion` reads raw files, `src/validation` checks them, `src/preprocessing` cleans them, `src/features` engineers columns, `src/training` fits models, `src/evaluation` scores the champion, `src/prediction` serves predictions. Don't cross those lines — if you find yourself importing pandas plotting into `src/ingestion`, it belongs elsewhere.
- **No leakage.** Any panel feature (lag, rolling, trend, volatility) must be grouped by `iso3` and use `.shift(1)` before computing a rolling statistic, so a country's current-year features never see its current-year target.
- **Every module is independently runnable.** Each `src/**/*.py` file has an `if __name__ == "__main__":` block so a contributor can run and sanity-check it in isolation before wiring it into `main.py`.
- **Tests accompany features.** New feature-engineering functions, validation rules, or training helpers should ship with a corresponding test in `tests/`.

## Submitting changes

1. Create a branch off `main`: `git checkout -b feature/<short-description>`
2. Make your change, add/update tests, run `pytest tests/ -v` and confirm `python main.py --stage all` still completes cleanly.
3. Update `config/config.yaml` and `README.md` if you changed a default or added a new stage.
4. Open a pull request describing the business or technical motivation, not just the diff.

## Reporting issues

Open a GitHub issue with: what you ran, what you expected, what happened instead, and the relevant lines from `logs/pipeline.log`.
