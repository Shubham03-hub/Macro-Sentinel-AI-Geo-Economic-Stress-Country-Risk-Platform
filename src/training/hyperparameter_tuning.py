"""
Hyperparameter tuning wrappers. Supports grid or randomized search over the
grids defined in config.yaml, scored on negative MAE via K-fold CV. Kept
separate from train_model.py so tuning strategy can be swapped (e.g. to
Optuna) without touching the training orchestration.
"""

from typing import Any, Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
import lightgbm as lgb
import xgboost as xgb

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_search(estimator, param_grid: Dict, cfg: Dict, n_splits: int, seed: int):
    method = cfg["training"]["hyperparameter_search"]["method"]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    common_kwargs = dict(
        estimator=estimator,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    if method == "grid":
        return GridSearchCV(param_grid=param_grid, **common_kwargs)

    n_iter = cfg["training"]["hyperparameter_search"]["n_iter_random"]
    return RandomizedSearchCV(
        param_distributions=param_grid, n_iter=n_iter, random_state=seed, **common_kwargs
    )


def tune_random_forest(X_train, y_train, config: Dict = None) -> Tuple[Any, Dict]:
    cfg = config or load_config()
    seed = cfg["project"]["random_seed"]
    grid = cfg["training"]["random_forest_grid"]

    base = RandomForestRegressor(random_state=seed, n_jobs=-1)
    search = _make_search(base, grid, cfg, cfg["training"]["n_splits_cv"], seed)
    logger.info("Tuning RandomForestRegressor...")
    search.fit(X_train, y_train)
    logger.info(f"Best RF params: {search.best_params_} | CV MAE: {-search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_


def tune_xgboost(X_train, y_train, config: Dict = None) -> Tuple[Any, Dict]:
    cfg = config or load_config()
    seed = cfg["project"]["random_seed"]
    grid = cfg["training"]["xgboost_grid"]

    base = xgb.XGBRegressor(
        random_state=seed, n_jobs=-1, objective="reg:squarederror", verbosity=0
    )
    search = _make_search(base, grid, cfg, cfg["training"]["n_splits_cv"], seed)
    logger.info("Tuning XGBRegressor...")
    search.fit(X_train, y_train)
    logger.info(f"Best XGB params: {search.best_params_} | CV MAE: {-search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_


def tune_lightgbm(X_train, y_train, config: Dict = None) -> Tuple[Any, Dict]:
    cfg = config or load_config()
    seed = cfg["project"]["random_seed"]
    grid = cfg["training"]["lightgbm_grid"]

    base = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1)
    search = _make_search(base, grid, cfg, cfg["training"]["n_splits_cv"], seed)
    logger.info("Tuning LGBMRegressor...")
    search.fit(X_train, y_train)
    logger.info(f"Best LGBM params: {search.best_params_} | CV MAE: {-search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_
