"""Task 9 (Phase 1): Dataset Mixing Optimization (Level-1 allocation only)."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.allocation import evaluate_allocation, normalize_mixture
from ..models import Sample
from .base import Task


class DatasetMixingTask(Task):
    name = "dataset_mixing"
    seed_filename = "dataset_mixing.jsonl"
    required_keys = ["ratios"]
    system_prompt = (
        "You optimize training data mixture ratios under a fixed sample budget. "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        domains = sample.input.get("domains") or []
        budget = sample.input.get("budget", 100000)
        target = sample.input.get("target", "")
        domain_lines = "\n".join(f"- {d}" for d in domains)
        return (
            "Propose mixture percentages for the domains below.\n"
            "Return JSON: {\"ratios\": {\"domain\": percent, ...}} with percentages summing to 100.\n\n"
            f"Target: {target}\n"
            f"Budget: {budget} samples\n"
            f"Domains:\n{domain_lines}\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        mixture = normalize_mixture(raw)
        if mixture is None:
            raise ValueError(f"invalid ratios: {raw!r}")
        return {"ratios": mixture}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        enriched = []
        for row in records:
            label = dict(row.get("label") or {})
            # domains may live on sample input mirrored into record
            enriched.append(row if "domains" in label or "reference" in label else row)
        return evaluate_allocation(enriched)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        ref = sample.label.get("reference") or {}
        if ref:
            return {"ratios": {k: float(v) for k, v in ref.items()}}
        domains = sample.input.get("domains") or []
        if not domains:
            return {"ratios": {}}
        share = round(100.0 / len(domains), 4)
        ratios = {d: share for d in domains}
        # fix rounding
        total = sum(ratios.values())
        if domains and abs(total - 100.0) > 1e-6:
            ratios[domains[0]] = round(ratios[domains[0]] + (100.0 - total), 4)
        return {"ratios": ratios}
