"""Task 11: Data Utility Prediction."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.utility import evaluate_utility_prediction
from ..models import Sample
from .base import Task


class DataUtilityPredictionTask(Task):
    name = "data_utility_prediction"
    seed_filename = "data_utility_prediction.jsonl"
    required_keys = ["utility"]
    system_prompt = (
        "You estimate how useful a sample is for LLM training. "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        text = sample.input.get("text") or sample.input.get("sample") or ""
        return (
            "Predict the training utility of this sample on a 0-5 scale "
            "(0 = useless / harmful, 5 = highly useful educational content).\n"
            'Return JSON: {"utility": number}\n\n'
            f"Sample:\n{text}\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        util = raw.get("utility", raw.get("score"))
        try:
            value = float(util)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid utility: {util!r}") from exc
        return {"utility": value}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_utility_prediction(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        util = sample.label.get("utility", sample.label.get("score", 3.0))
        return {"utility": float(util)}
