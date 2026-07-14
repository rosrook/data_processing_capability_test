"""Load multi-model specs from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from .errors import LLMConfigError
from .llm_client import LLMConfig


def load_model_configs(path: Path) -> List[LLMConfig]:
    if not path.is_file():
        raise LLMConfigError(f"models file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("models")
    if not isinstance(rows, list) or not rows:
        raise LLMConfigError(f"models file has empty models list: {path}")
    configs: List[LLMConfig] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise LLMConfigError("each model entry must be a mapping")
        cfg = LLMConfig.from_mapping(row)
        if cfg.name in seen:
            raise LLMConfigError(f"duplicate model name: {cfg.name}")
        seen.add(cfg.name)
        configs.append(cfg)
    return configs
