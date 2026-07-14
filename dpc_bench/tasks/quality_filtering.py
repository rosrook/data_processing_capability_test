"""Task 2: Quality Filtering."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.classification import binary_classification_metrics, normalize_keep_remove
from ..models import Sample
from .base import Task


class QualityFilteringTask(Task):
    name = "quality_filtering"
    seed_filename = "quality_filtering.jsonl"
    required_keys = ["decision"]
    system_prompt = (
        "You are a pretraining data quality filter. Decide whether a text should be kept "
        "for an LLM training corpus. Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        text = sample.input.get("text", "")
        return (
            "Decide whether to keep this text in a high-quality training set.\n"
            "Return JSON: {\"decision\": \"keep\"} or {\"decision\": \"remove\"}.\n\n"
            f"Text:\n{text}\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        decision = normalize_keep_remove(raw.get("decision"))
        if decision is None:
            # allow keep/remove under alternate key
            decision = normalize_keep_remove(raw.get("label"))
        if decision is None:
            raise ValueError(f"invalid decision value: {raw.get('decision')!r}")
        out: Dict[str, Any] = {"decision": decision}
        if "score" in raw:
            try:
                out["score"] = float(raw["score"])
            except (TypeError, ValueError):
                pass
        return out

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true = []
        y_pred = []
        for row in records:
            gold = normalize_keep_remove((row.get("label") or {}).get("decision"))
            y_true.append(gold or "remove")
            pred = row.get("prediction") or {}
            if row.get("error") or not pred:
                y_pred.append(None)
            else:
                y_pred.append(normalize_keep_remove(pred.get("decision")))
        return binary_classification_metrics(y_true, y_pred, positive="keep")

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        return {"decision": normalize_keep_remove(sample.label.get("decision")) or "remove"}
