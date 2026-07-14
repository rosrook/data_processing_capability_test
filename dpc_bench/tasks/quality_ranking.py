"""Task 4: Quality Ranking."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.ranking import evaluate_ranking, normalize_id_list
from ..models import Sample
from .base import Task


class QualityRankingTask(Task):
    name = "quality_ranking"
    seed_filename = "quality_ranking.jsonl"
    required_keys = ["ranking"]
    system_prompt = (
        "You rank pretraining/text samples by training quality (best first). "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        cands = sample.input.get("candidates") or []
        blocks = []
        for c in cands:
            cid = c.get("id")
            text = c.get("text", "")
            blocks.append(f"[{cid}]\n{text}\n")
        return (
            "Rank the candidates from highest to lowest quality for LLM pretraining.\n"
            'Return JSON: {"ranking": ["id1", "id2", ...]}\n\n'
            + "\n".join(blocks)
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        order = normalize_id_list(raw)
        if not order:
            raise ValueError(f"invalid ranking: {raw!r}")
        return {"ranking": order}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_ranking(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        order = sample.label.get("ranking") or sample.label.get("order") or []
        return {"ranking": list(order)}
