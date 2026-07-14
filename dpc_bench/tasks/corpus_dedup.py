"""Corpus Deduplication: remove near-duplicates from a document batch."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.corpus import evaluate_corpus_dedup
from ..metrics.ranking import normalize_id_list
from ..models import Sample
from .base import Task


class CorpusDedupTask(Task):
    name = "corpus_dedup"
    seed_filename = "corpus_dedup.jsonl"
    required_keys = ["remove"]
    system_prompt = (
        "You deduplicate a document corpus: drop near-duplicate / paraphrase copies, "
        "keeping one representative per duplicate group. Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        docs = sample.input.get("documents") or []
        blocks = []
        for d in docs:
            blocks.append(f"[{d.get('id')}]\n{d.get('text', '')}\n")
        return (
            "Identify near-duplicate documents in the corpus and list IDs to REMOVE "
            "(keep one survivor per duplicate cluster; unique docs should not be removed).\n"
            'Return JSON: {"remove": ["id1", "id2", ...]}\n'
            "If there are no duplicates, return {\"remove\": []}.\n\n"
            + "\n".join(blocks)
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(raw, dict) and "remove" in raw:
            remove = normalize_id_list(raw.get("remove"))
            if remove is None:
                remove = []
            return {"remove": remove}
        keep = normalize_id_list(raw.get("keep") if isinstance(raw, dict) else raw)
        if keep is not None:
            return {"keep": keep}
        remove = normalize_id_list(raw)
        if remove is None:
            raise ValueError(f"invalid remove/keep list: {raw!r}")
        return {"remove": remove}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_corpus_dedup(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        remove = sample.label.get("remove") or []
        return {"remove": list(remove)}
