"""Task 5: Diversity Selection."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.ranking import normalize_id_list
from ..metrics.selection import evaluate_diversity_selection
from ..models import Sample
from .base import Task


class DiversitySelectionTask(Task):
    name = "diversity_selection"
    seed_filename = "diversity_selection.jsonl"
    required_keys = ["selected"]
    system_prompt = (
        "You select a diverse subset of instruction samples under a fixed budget. "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        cands = sample.input.get("candidates") or []
        budget = sample.input.get("budget", 0)
        blocks = []
        for c in cands:
            blocks.append(
                f"[{c.get('id')}] topic={c.get('topic') or c.get('cluster')}\n"
                f"{c.get('text') or c.get('instruction') or ''}\n"
            )
        return (
            f"Select exactly {budget} samples that maximize topical diversity.\n"
            'Return JSON: {"selected": ["id1", "id2", ...]}\n\n'
            + "\n".join(blocks)
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        selected = normalize_id_list(raw.get("selected") if isinstance(raw, dict) else raw)
        if not selected:
            selected = normalize_id_list(raw)
        if not selected:
            raise ValueError(f"invalid selected: {raw!r}")
        return {"selected": selected}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_diversity_selection(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        selected = sample.label.get("selected") or sample.label.get("reference") or []
        return {"selected": list(selected)}
