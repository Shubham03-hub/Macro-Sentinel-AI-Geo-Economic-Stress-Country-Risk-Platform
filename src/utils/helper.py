"""
Small, dependency-free helpers shared across pipeline stages: config loading,
project-root-relative path resolution, and the risk-category lookup used by
both the prediction module and the dashboard.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and return the project YAML config as a dict."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root, creating parent dirs if needed."""
    path = PROJECT_ROOT / relative_path
    return path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def categorize_risk(score: float, config: Dict[str, Any]) -> str:
    """Map a numeric stress score (0-100) to a business-facing risk category."""
    categories = config["risk_scoring"]["categories"]
    for cat in categories:
        if cat["min"] <= score < cat["max"]:
            return cat["label"]
    return categories[-1]["label"]


def risk_category_color(label: str) -> str:
    """Consistent color coding for the dashboard and reports."""
    mapping = {
        "Low": "#2E7D32",
        "Moderate": "#F9A825",
        "Elevated": "#EF6C00",
        "High": "#D84315",
        "Severe": "#B71C1C",
    }
    return mapping.get(label, "#616161")
