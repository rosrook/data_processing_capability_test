"""Task 1: Semantic Deduplication."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.classification import binary_classification_metrics, normalize_yes_no
from ..models import Sample
from .base import Task


class SemanticDedupTask(Task):
    name = "semantic_dedup"
    seed_filename = "semantic_dedup.jsonl"
    required_keys = ["duplicate"]
    system_prompt = (
        "You are a data cleaning expert. Decide whether two texts are semantic duplicates "
        "(same core meaning, including paraphrases). Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        a = sample.input.get("text_a", "")
        b = sample.input.get("text_b", "")
        return (
            "Judge if Text A and Text B are semantic duplicates.\n"
            "Return JSON: {\"duplicate\": \"YES\"} or {\"duplicate\": \"NO\"}.\n\n"
            f"Text A:\n{a}\n\nText B:\n{b}\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        label = normalize_yes_no(raw.get("duplicate"))
        if label is None:
            raise ValueError(f"invalid duplicate value: {raw.get('duplicate')!r}")
        return {"duplicate": label}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true = []
        y_pred = []
        for row in records:
            gold = normalize_yes_no((row.get("label") or {}).get("duplicate"))
            y_true.append(gold or "NO")
            pred = row.get("prediction") or {}
            if row.get("error") or not pred:
                y_pred.append(None)
            else:
                y_pred.append(normalize_yes_no(pred.get("duplicate")))
        metrics = binary_classification_metrics(y_true, y_pred, positive="YES")
        # Hard-negative subset accuracy if annotated
        hard_correct = hard_total = 0
        for row in records:
            meta = row.get("meta") or {}
            if meta.get("hard_negative"):
                hard_total += 1
                gold = normalize_yes_no((row.get("label") or {}).get("duplicate"))
                pred = normalize_yes_no((row.get("prediction") or {}).get("duplicate"))
                if gold is not None and pred == gold:
                    hard_correct += 1
        if hard_total:
            metrics["hard_negative_accuracy"] = round(hard_correct / hard_total, 6)
            metrics["hard_negative_n"] = hard_total
        return metrics

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        return {"duplicate": normalize_yes_no(sample.label.get("duplicate")) or "NO"}
