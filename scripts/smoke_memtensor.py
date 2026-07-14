#!/usr/bin/env python3
"""Smoke-test memtensor Chat Completions using dpc_bench LLMConfig/LLMClient.

Requires .env:
  MODEL_gpt_memtensor_URL
  MODEL_gpt_memtensor_KEY
  GPT_API_MODEL (optional)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpc_bench.env_loader import load_dotenv_once
from dpc_bench.llm_client import LLMClient, LLMConfig


def main() -> int:
    loaded = load_dotenv_once()
    print(f"env loaded from: {loaded}")
    cfg = LLMConfig.from_env(name="gpt-memtensor")
    print(f"backend={cfg.backend} model={cfg.model}")
    print(f"base_url={cfg.base_url}")
    client = LLMClient(cfg)
    # Plain chat via openai_compat path (JSON-ish instruction)
    text = client._chat_openai_compat(
        "You are a concise assistant.",
        "Reply with exactly: memtensor-ok",
        max_completion_tokens=32,
        temperature=0.0,
    )
    print(f"assistant: {text!r}")
    print(f"calls: {client.call_count}")
    ok = "memtensor-ok" in (text or "").lower().replace(" ", "")
    print("status:", "OK" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
