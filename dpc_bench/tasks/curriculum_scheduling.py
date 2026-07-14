"""Task 10: Curriculum Scheduling."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.curriculum import evaluate_curriculum, normalize_phases
from ..metrics.ranking import normalize_id_list
from ..models import Sample
from .base import Task


class CurriculumSchedulingTask(Task):
    name = "curriculum_scheduling"
    seed_filename = "curriculum_scheduling.jsonl"
    required_keys = ["schedule"]
    system_prompt = (
        "You design a curriculum order for training. "
        "Prefer easy → medium → hard unless evidence suggests otherwise. "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        phases = sample.input.get("phases") or ["easy", "medium", "hard"]
        steps = sample.input.get("training_steps") or 1000
        notes = sample.input.get("notes") or ""
        items = sample.input.get("items") or []
        if items:
            lines = "\n".join(
                f"- {it.get('id')}: difficulty={it.get('difficulty')}, "
                f"text={str(it.get('text') or '')[:160]}"
                for it in items
            )
            return (
                f"Order the items for {steps} training steps to maximize learning stability.\n"
                'Return JSON: {"schedule": ["id1", "id2", ...]}\n\n'
                f"Items:\n{lines}\n"
            )
        return (
            f"Choose a curriculum phase order for {steps} training steps.\n"
            f"Available phases: {phases}\n"
            f"Notes: {notes}\n"
            'Return JSON: {"schedule": ["easy", "medium", "hard"]}\n'
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        schedule = normalize_phases(raw) or normalize_id_list(raw)
        if not schedule:
            raise ValueError(f"invalid schedule: {raw!r}")
        return {"schedule": schedule}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_curriculum(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        gold = sample.label.get("schedule") or sample.label.get("order") or [
            "easy",
            "medium",
            "hard",
        ]
        return {"schedule": list(gold)}
