"""Multi-model / multi-task evaluation runner."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

from .io import read_json, read_jsonl, write_json
from .llm_client import ChatClient, DryRunClient, LLMClient, LLMConfig
from .logging_util import get_logger, setup_logging
from .models import PredictionRecord, Sample, TaskRunResult
from .paths import DEFAULT_OUTPUT_DIR, DEFAULT_SEED_DIR
from .tasks import get_task, list_tasks
from .tasks.base import Task

log = get_logger(__name__)


def _safe_dirname(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "model"


@dataclass
class RunConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed_dir: Path = DEFAULT_SEED_DIR
    tasks: Optional[List[str]] = None
    limit: Optional[int] = None
    dry_run: bool = False
    continue_on_error: bool = True
    max_completion_tokens: int = 1024
    max_retries: int = 1
    temperature: float = 0.0
    resume: bool = True  # skip sample ids already in predictions.jsonl
    jobs: int = 1  # parallel (model, task) workers
    # process = ProcessPool; thread = ThreadPool (safer under some sandbox/limits)
    executor: str = "process"
    verbose: bool = False


def resolve_task_names(task: str) -> List[str]:
    if task.strip().lower() in {"all", "*"}:
        return list_tasks()
    names = [p.strip() for p in task.split(",") if p.strip()]
    for name in names:
        get_task(name)  # validate
    return names


def _task_paths(model_cfg: LLMConfig, task_name: str, config: RunConfig) -> Tuple[Path, Path, Path]:
    task = get_task(task_name)
    out_dir = config.output_dir / _safe_dirname(model_cfg.name) / task.name
    return out_dir, out_dir / "predictions.jsonl", out_dir / "metrics.json"


def _load_completed_result(
    model_cfg: LLMConfig,
    task_name: str,
    config: RunConfig,
) -> Optional[TaskRunResult]:
    """If resume and all samples already present + metrics.json, skip the task."""
    if not config.resume:
        return None
    task = get_task(task_name)
    samples = task.load_samples(seed_dir=config.seed_dir, limit=config.limit)
    _, pred_path, metrics_path = _task_paths(model_cfg, task_name, config)
    if not pred_path.is_file() or not metrics_path.is_file():
        return None
    existing = read_jsonl(pred_path)
    done_ids = {str(row.get("id") or "") for row in existing if row.get("id")}
    if not samples or any(s.id not in done_ids for s in samples):
        return None
    try:
        payload = read_json(metrics_path)
    except Exception:
        return None
    metrics = payload.get("metrics") or {}
    log.info(
        "skip complete model=%s task=%s n=%d",
        model_cfg.name,
        task.name,
        len(samples),
    )
    return TaskRunResult(
        model=model_cfg.name,
        task=task.name,
        n=int(metrics.get("n") or len(existing)),
        metrics=metrics,
        predictions_path=str(pred_path),
        metrics_path=str(metrics_path),
        parse_errors=int(payload.get("parse_errors") or 0),
        call_errors=int(payload.get("call_errors") or 0),
    )


def _run_job_inline(
    model_cfg: LLMConfig,
    task_name: str,
    config: RunConfig,
) -> TaskRunResult:
    cached = _load_completed_result(model_cfg, task_name, config)
    if cached is not None:
        return cached
    return run_task_for_model(model_cfg, task_name, config)


def _run_job_worker(
    model_cfg: LLMConfig,
    task_name: str,
    config: RunConfig,
) -> TaskRunResult:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    setup_logging(
        verbose=config.verbose,
        output_dir=config.output_dir,
        command=f"job_{_safe_dirname(model_cfg.name)}_{task_name}",
    )
    return _run_job_inline(model_cfg, task_name, config)


def run_task_for_model(
    model_cfg: LLMConfig,
    task_name: str,
    config: RunConfig,
    *,
    client: Optional[ChatClient] = None,
) -> TaskRunResult:
    task = get_task(task_name)
    samples = task.load_samples(seed_dir=config.seed_dir, limit=config.limit)
    out_dir = config.output_dir / _safe_dirname(model_cfg.name) / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    if client is None:
        if config.dry_run:
            client = DryRunClient(lambda _user: {})  # replaced per-sample below
        else:
            client = LLMClient(model_cfg)

    done_ids: set[str] = set()
    records: List[Dict[str, Any]] = []
    if config.resume and pred_path.is_file():
        existing = read_jsonl(pred_path)
        for row in existing:
            rid = str(row.get("id") or "")
            if rid:
                done_ids.add(rid)
                records.append(row)
        if done_ids:
            log.info(
                "resume model=%s task=%s already_have=%d",
                model_cfg.name,
                task.name,
                len(done_ids),
            )

    parse_errors = sum(
        1
        for r in records
        if r.get("error") and ("parse" in str(r.get("error")).lower() or r.get("prediction") is None)
    )
    call_errors = sum(
        1 for r in records if r.get("error") and not ("parse" in str(r.get("error")).lower())
    )

    # append mode for incremental durability
    append_fh = pred_path.open("a" if done_ids else "w", encoding="utf-8")
    try:
        if not done_ids:
            # truncate handled by "w"
            pass
        for sample in samples:
            if sample.id in done_ids:
                continue
            record = _predict_one(
                client=client,
                model_cfg=model_cfg,
                task=task,
                sample=sample,
                config=config,
            )
            row = record.to_dict()
            if record.error:
                if "parse" in (record.error or "").lower() or record.prediction is None:
                    parse_errors += 1
                else:
                    call_errors += 1
            records.append(row)
            append_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            append_fh.flush()
            if len(records) % 20 == 0:
                log.info(
                    "progress model=%s task=%s done=%d/%d",
                    model_cfg.name,
                    task.name,
                    len(records),
                    len(samples),
                )
    finally:
        append_fh.close()

    metrics = task.evaluate(records)
    metrics_payload = {
        "model": model_cfg.name,
        "model_id": model_cfg.model,
        "backend": model_cfg.backend,
        "task": task.name,
        "limit": config.limit,
        "dry_run": config.dry_run,
        "metrics": metrics,
        "parse_errors": parse_errors,
        "call_errors": call_errors,
    }
    write_json(metrics_path, metrics_payload)
    log.info(
        "model=%s task=%s n=%d primary=%s=%.4f parse_errors=%d",
        model_cfg.name,
        task.name,
        metrics.get("n", len(records)),
        metrics.get("primary_name", "primary"),
        float(metrics.get("primary") or 0.0),
        parse_errors,
    )
    return TaskRunResult(
        model=model_cfg.name,
        task=task.name,
        n=int(metrics.get("n") or len(records)),
        metrics=metrics,
        predictions_path=str(pred_path),
        metrics_path=str(metrics_path),
        parse_errors=parse_errors,
        call_errors=call_errors,
    )


def _predict_one(
    *,
    client: ChatClient,
    model_cfg: LLMConfig,
    task: Task,
    sample: Sample,
    config: RunConfig,
) -> PredictionRecord:
    meta = dict(sample.meta)
    if "constraints" in sample.input:
        meta["requires_reasoning"] = bool(
            (sample.input.get("constraints") or {}).get("requires_reasoning")
        )
        meta["topic"] = sample.input.get("topic")
    label = dict(sample.label)
    if "domains" in sample.input and "domains" not in label:
        label["domains"] = sample.input["domains"]
    if "constraints" in sample.input:
        label.setdefault("constraints", sample.input["constraints"])

    base = PredictionRecord(
        id=sample.id,
        task=task.name,
        model=model_cfg.name,
        input=dict(sample.input),
        label=label,
        prediction=None,
        raw=None,
        error=None,
        meta=meta,
    )

    try:
        if config.dry_run:
            raw = task.dry_run_prediction(sample)
        else:
            raw = client.chat_json(
                task.system_prompt,
                task.build_user_prompt(sample),
                required_keys=task.required_keys,
                max_completion_tokens=config.max_completion_tokens,
                max_retries=config.max_retries,
                temperature=config.temperature,
            )
        base.raw = raw
        base.prediction = task.parse_prediction(raw)
    except Exception as exc:
        base.error = f"{type(exc).__name__}: {exc}"
        if not config.continue_on_error:
            raise
        log.warning("sample %s failed: %s", sample.id, base.error)
    return base


def run_benchmark(
    model_configs: Sequence[LLMConfig],
    config: RunConfig,
) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    task_names = config.tasks or list_tasks()
    jobs = max(1, int(config.jobs or 1))
    work: List[Tuple[LLMConfig, str]] = [
        (model_cfg, task_name)
        for model_cfg in model_configs
        for task_name in task_names
    ]
    log.info(
        "benchmark models=%d tasks=%d units=%d jobs=%d resume=%s",
        len(model_configs),
        len(task_names),
        len(work),
        jobs,
        config.resume,
    )

    results: List[TaskRunResult] = []
    if jobs == 1:
        for model_cfg, task_name in work:
            cached = _load_completed_result(model_cfg, task_name, config)
            if cached is not None:
                results.append(cached)
            else:
                results.append(run_task_for_model(model_cfg, task_name, config))
    else:
        # One worker per (model, task): separate predictions.jsonl paths, safe under resume.
        mode = (config.executor or "process").strip().lower()
        if mode not in {"process", "thread"}:
            mode = "process"
        pool_cls = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
        worker = _run_job_worker if mode == "process" else _run_job_inline
        pending: Dict[Any, Tuple[str, str]] = {}
        try:
            pool_cm = pool_cls(max_workers=jobs)
        except (PermissionError, NotImplementedError, OSError) as exc:
            log.warning(
                "ProcessPool unavailable (%s); falling back to ThreadPoolExecutor",
                exc,
            )
            pool_cm = ThreadPoolExecutor(max_workers=jobs)
            worker = _run_job_inline
            mode = "thread"
        log.info("parallel executor=%s workers=%d", mode, jobs)
        with pool_cm as pool:
            for model_cfg, task_name in work:
                fut = pool.submit(worker, model_cfg, task_name, config)
                pending[fut] = (model_cfg.name, task_name)
            for fut in as_completed(pending):
                model_name, task_name = pending[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    log.info(
                        "finished model=%s task=%s n=%d",
                        model_name,
                        task_name,
                        result.n,
                    )
                except Exception as exc:
                    log.error(
                        "worker failed model=%s task=%s: %s",
                        model_name,
                        task_name,
                        exc,
                    )
                    if not config.continue_on_error:
                        raise
        # Stable order for summary
        order = {(m.name, t): i for i, (m, t) in enumerate(work)}
        results.sort(key=lambda r: order.get((r.model, r.task), 10**9))

    summary = {
        "output_dir": str(config.output_dir),
        "tasks": task_names,
        "models": [m.name for m in model_configs],
        "limit": config.limit,
        "dry_run": config.dry_run,
        "jobs": jobs,
        "resume": config.resume,
        "results": [r.to_dict() for r in results],
    }
    write_json(config.output_dir / "summary.json", summary)
    (config.output_dir / "comparison.md").write_text(
        build_comparison(config.output_dir), encoding="utf-8"
    )
    return summary


def build_comparison(output_dir: Path) -> str:
    """Scan */*/metrics.json under output_dir and build a markdown table."""
    rows: List[Dict[str, Any]] = []
    if not output_dir.is_dir():
        return "# Comparison\n\nNo results.\n"

    for metrics_path in sorted(output_dir.glob("*/*/metrics.json")):
        try:
            from .io import read_json

            payload = read_json(metrics_path)
        except Exception:
            continue
        metrics = payload.get("metrics") or {}
        rows.append(
            {
                "model": payload.get("model") or metrics_path.parent.parent.name,
                "task": payload.get("task") or metrics_path.parent.name,
                "primary_name": metrics.get("primary_name", "primary"),
                "primary": metrics.get("primary"),
                "n": metrics.get("n"),
                "extra": metrics,
            }
        )

    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    cell: Dict[tuple, Dict[str, Any]] = {(r["model"], r["task"]): r for r in rows}

    lines = [
        "# Data Processing Capability — Comparison",
        "",
        "Primary metrics (higher is better unless noted).",
        "",
    ]
    header = "| Model | " + " | ".join(tasks) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(tasks)) + " |"
    lines.extend([header, sep])
    for model in models:
        cells = []
        for task in tasks:
            item = cell.get((model, task))
            if not item or item.get("primary") is None:
                cells.append("-")
            else:
                name = item.get("primary_name") or "primary"
                val = item["primary"]
                cells.append(f"{val:.4f} ({name})")
        lines.append("| " + model + " | " + " | ".join(cells) + " |")

    lines.extend(["", "## Details", ""])
    for r in rows:
        m = r["extra"]
        detail_bits = []
        for key in ("accuracy", "f1", "valid_rate", "mean_l1", "mean_score", "hard_negative_accuracy"):
            if key in m and m[key] is not None:
                detail_bits.append(f"{key}={m[key]}")
        lines.append(
            f"- **{r['model']} / {r['task']}** (n={r['n']}): " + ", ".join(detail_bits)
        )
    lines.append("")
    return "\n".join(lines)


def compare_only(output_dir: Path) -> Path:
    md = build_comparison(output_dir)
    path = output_dir / "comparison.md"
    path.write_text(md, encoding="utf-8")
    return path
