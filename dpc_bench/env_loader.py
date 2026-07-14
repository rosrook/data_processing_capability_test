"""Minimal .env loader without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
_WORKSPACE_ROOT = _REPO_ROOT.parent

DEFAULT_SEARCH_PATHS = (
    _REPO_ROOT / ".env",
    _PACKAGE_DIR / ".env",
    _WORKSPACE_ROOT / "decision_rgu" / ".env",
    _WORKSPACE_ROOT / "trace_rgu_tool_v2" / ".env",
    _WORKSPACE_ROOT / "rubrics_distiller" / ".env",
)

_LOADED_PATH: Optional[Path] = None
_LOAD_TRIED = False


def _parse_line(line: str) -> Optional[tuple[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export ") :].lstrip()
    if "=" not in s:
        return None
    key, _, raw_value = s.partition("=")
    key = key.strip()
    value = raw_value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_dotenv_once(
    search_paths: Iterable[Path] = DEFAULT_SEARCH_PATHS,
    overwrite: bool = False,
) -> Optional[Path]:
    global _LOADED_PATH, _LOAD_TRIED
    if _LOAD_TRIED:
        return _LOADED_PATH
    _LOAD_TRIED = True

    for candidate in search_paths:
        if not candidate.is_file():
            continue
        with candidate.open("r", encoding="utf-8") as f:
            for raw in f:
                kv = _parse_line(raw)
                if kv is None:
                    continue
                key, value = kv
                if not overwrite and key in os.environ:
                    continue
                os.environ[key] = value
        _LOADED_PATH = candidate
        break
    return _LOADED_PATH
