"""
Centralized logger factory. Every pipeline module calls get_logger(__name__)
instead of configuring logging ad hoc, so log format/level stay consistent
and everything lands in logs/pipeline.log as well as stdout.
"""

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def _configure_root(log_file: str = "logs/pipeline.log", level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, log_file: str = "logs/pipeline.log", level: str = "INFO") -> logging.Logger:
    """Return a module-scoped logger, configuring root handlers on first call."""
    _configure_root(log_file=log_file, level=level)
    return logging.getLogger(name)
