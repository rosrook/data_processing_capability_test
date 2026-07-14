#!/usr/bin/env python3
"""Backward-compatible wrapper: build the 7 newer tasks from HuggingFace.

Preferred entrypoint (all 11 tasks):

  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/prepare_seed_from_hf.py --tasks all --limit-per-task 400

This wrapper only rebuilds the post-Phase-1 tasks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "scripts" / "prepare_seed_from_hf.py"

NEW_TASKS = ",".join(
    [
        "corpus_filtering",
        "format_repair",
        "quality_ranking",
        "diversity_selection",
        "reasoning_generation",
        "hard_sample_generation",
        "curriculum_scheduling",
        "data_utility_prediction",
        "corpus_dedup",
    ]
)


def main() -> int:
    # Preserve historic CLI: --limit-per-task / --seed / --hf-endpoint
    argv = [sys.executable, str(UNIFIED), "--tasks", NEW_TASKS, *sys.argv[1:]]
    print("delegating to:", " ".join(argv))
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
