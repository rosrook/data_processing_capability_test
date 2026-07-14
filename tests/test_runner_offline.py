from pathlib import Path

from dpc_bench.llm_client import LLMConfig
from dpc_bench.runner import RunConfig, build_comparison, run_benchmark
from dpc_bench.tasks import list_tasks


def test_dry_run_all_tasks(tmp_path: Path):
    models = [
        LLMConfig(name="mock-a", base_url="http://x", api_key="k", model="m"),
        LLMConfig(name="mock-b", base_url="http://x", api_key="k", model="m"),
    ]
    config = RunConfig(
        output_dir=tmp_path / "out",
        dry_run=True,
        limit=5,
        tasks=list_tasks(),
    )
    summary = run_benchmark(models, config)
    assert len(summary["results"]) == len(models) * len(list_tasks())
    cmp_path = tmp_path / "out" / "comparison.md"
    assert cmp_path.is_file()
    text = cmp_path.read_text(encoding="utf-8")
    assert "mock-a" in text
    assert "semantic_dedup" in text

    # primary scores should be strong under dry-run oracle labels
    by_key = {(r["model"], r["task"]): r["metrics"] for r in summary["results"]}
    assert by_key[("mock-a", "semantic_dedup")]["f1"] == 1.0
    assert by_key[("mock-a", "quality_filtering")]["f1"] == 1.0
    assert by_key[("mock-a", "dataset_mixing")]["mean_score"] == 1.0
    assert by_key[("mock-a", "instruction_generation")]["mean_score"] >= 0.9


def test_build_comparison_empty(tmp_path: Path):
    md = build_comparison(tmp_path)
    assert "No results" in md or "Comparison" in md
