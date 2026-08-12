"""
Thin wrapper around joblib for persisting/loading trained models and the
fitted preprocessing artifacts (scaler, feature list) they depend on. Keeping
this in one place means training, prediction, and the dashboard all load
models the exact same way.
"""

from pathlib import Path
from typing import Any, Dict

import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved artifact to {path}")


def load_artifact(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found at {path}. Run the training pipeline first: "
            f"python main.py --stage train"
        )
    logger.info(f"Loading artifact from {path}")
    return joblib.load(path)


def load_champion_bundle(models_dir: Path) -> Dict[str, Any]:
    """
    Load everything the prediction pipeline / dashboard need in one call:
    the champion model, the fitted scaler, and the ordered feature list.
    """
    bundle = {
        "model": load_artifact(models_dir / "champion_model.joblib"),
        "scaler": load_artifact(models_dir / "feature_scaler.joblib"),
        "feature_columns": load_artifact(models_dir / "feature_columns.joblib"),
        "metadata": load_artifact(models_dir / "champion_metadata.joblib"),
    }
    return bundle
