"""CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .env_loader import load_dotenv_once
from .errors import LLMConfigError
from .llm_client import LLMConfig
from .logging_util import get_logger, setup_logging
from .model_registry import load_model_configs
from .paths import DEFAULT_OUTPUT_DIR, DEFAULT_SEED_DIR
from .runner import RunConfig, compare_only, resolve_task_names, run_benchmark
from .tasks import list_tasks

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpc-bench",
        description="Data Processing Capability Benchmark (multi-model Level-1 eval)",
    )
    parser.add_argument("--version", action="version", version=f"dpc-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run benchmarks for one or more models")
    run_p.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help="YAML file listing models (see models.example.yaml). If omitted, use .env single model.",
    )
    run_p.add_argument(
        "--task",
        default="all",
        help=f"Task name, comma list, or 'all'. Available: {', '.join(list_tasks())}",
    )
    run_p.add_argument("--limit", type=int, default=None, help="Max samples per task")
    run_p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_p.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    run_p.add_argument("--dry-run", action="store_true", help="Use fixture labels; no API calls")
    run_p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first sample error (default: continue)",
    )
    run_p.add_argument("--max-completion-tokens", type=int, default=1024)
    run_p.add_argument("--max-retries", type=int, default=2)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing predictions.jsonl and overwrite",
    )
    run_p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for (model, task) units (default: 1). Resume-safe.",
    )
    run_p.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Parallel backend: process (default) or thread. Falls back to thread if process pool is unavailable.",
    )
    run_p.add_argument("-v", "--verbose", action="store_true")

    cmp = sub.add_parser("compare", help="Rebuild comparison.md from an output directory")
    cmp.add_argument("--output-dir", type=Path, required=True)
    cmp.add_argument("-v", "--verbose", action="store_true")

    lst = sub.add_parser("list-tasks", help="List available tasks")
    lst.add_argument("-v", "--verbose", action="store_true")
    return parser


def _stub_models_from_yaml(models_file: Path) -> List[LLMConfig]:
    import yaml

    data = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
    models: List[LLMConfig] = []
    for row in data.get("models") or []:
        models.append(
            LLMConfig(
                name=str(row.get("name") or "model"),
                base_url=str(row.get("base_url") or "http://localhost"),
                api_key="dry-run",
                model=str(row.get("model") or "dry-run"),
                backend=str(row.get("backend") or "openai_compat"),
            )
        )
    return models or [
        LLMConfig(name="dry-run", base_url="http://localhost", api_key="none", model="dry-run")
    ]


def _load_models(models_file: Optional[Path], *, dry_run: bool) -> List[LLMConfig]:
    load_dotenv_once()
    if models_file is None:
        if dry_run:
            return [
                LLMConfig(name="dry-run", base_url="http://localhost", api_key="none", model="dry-run")
            ]
        return [LLMConfig.from_env(name="default")]
    path = models_file.resolve()
    try:
        return load_model_configs(path)
    except LLMConfigError:
        if dry_run:
            return _stub_models_from_yaml(path)
        raise


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tasks":
        setup_logging(verbose=getattr(args, "verbose", False))
        for name in list_tasks():
            print(name)
        return 0

    if args.command == "compare":
        out = Path(args.output_dir).resolve()
        setup_logging(verbose=args.verbose, output_dir=out, command="compare")
        path = compare_only(out)
        print(path)
        return 0

    if args.command == "run":
        out = Path(args.output_dir).resolve()
        setup_logging(verbose=args.verbose, output_dir=out, command="run")
        try:
            models = _load_models(args.models_file, dry_run=bool(args.dry_run))
        except LLMConfigError as exc:
            log.error("%s", exc)
            return 2

        task_names = resolve_task_names(args.task)
        jobs = max(1, int(args.jobs or 1))
        config = RunConfig(
            output_dir=out,
            seed_dir=Path(args.seed_dir).resolve(),
            tasks=task_names,
            limit=args.limit,
            dry_run=bool(args.dry_run),
            continue_on_error=not bool(args.fail_fast),
            max_completion_tokens=args.max_completion_tokens,
            max_retries=args.max_retries,
            temperature=args.temperature,
            resume=not bool(args.no_resume),
            jobs=jobs,
            executor=str(args.executor),
            verbose=bool(args.verbose),
        )
        summary = run_benchmark(models, config)
        log.info(
            "done: %d model-task runs → %s",
            len(summary.get("results") or []),
            out / "comparison.md",
        )
        print(out / "comparison.md")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
