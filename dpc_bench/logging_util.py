"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
LOGGER_NAMESPACE = "dpc_bench"


def setup_logging(
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    command: Optional[str] = None,
) -> Optional[Path]:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    if output_dir is None:
        return None

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = (command or "session").replace("/", "_")
    log_path = log_dir / f"{timestamp}_{name}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path


def get_logger(name: str) -> logging.Logger:
    if name.startswith(LOGGER_NAMESPACE):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
