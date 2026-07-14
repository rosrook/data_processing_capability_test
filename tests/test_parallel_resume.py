"""Parallel (model, task) runs + resume skip."""

from __future__ import annotations

from pathlib import Path

from dpc_bench.llm_client import LLMConfig
from dpc_bench.runner import RunConfig, run_benchmark


def test_parallel_dry_run_and_resume(tmp_path: Path) -> None:
    seed_dir = Path(__file__).resolve().parents[1] / "data" / "seed"
    models = [
        LLMConfig(name="m1", base_url="http://x", api_key="k", model="m", backend="openai_compat"),
        LLMConfig(name="m2", base_url="http://x", api_key="k", model="m", backend="openai_compat"),
    ]
    config = RunConfig(
        output_dir=tmp_path / "out",
        seed_dir=seed_dir,
        tasks=["semantic_dedup", "dataset_mixing"],
        limit=3,
        dry_run=True,
        jobs=2,
        resume=True,
    )
    summary1 = run_benchmark(models, config)
    assert len(summary1["results"]) == 4
    assert (tmp_path / "out" / "comparison.md").is_file()

    # Second pass should skip completed units and still write summary.
    summary2 = run_benchmark(models, config)
    assert len(summary2["results"]) == 4
    for r in summary2["results"]:
        assert r["n"] >= 1
