"""Corpus Filtering: select keep-set from a dirty document batch (set recall)."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.corpus import evaluate_corpus_filtering
from ..metrics.ranking import normalize_id_list
from ..models import Sample
from .base import Task


class CorpusFilteringTask(Task):
    name = "corpus_filtering"
    seed_filename = "corpus_filtering.jsonl"
    required_keys = ["keep"]
    system_prompt = (
        "You filter a dirty pretraining corpus: keep high-quality educational documents "
        "and drop low-quality ones. Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        docs = sample.input.get("documents") or []
        budget = sample.input.get("budget")
        blocks = []
        for d in docs:
            blocks.append(f"[{d.get('id')}]\n{d.get('text', '')}\n")
        budget_line = (
            f"Optional soft budget: keep about {budget} documents.\n" if budget else ""
        )
        return (
            "From the dirty corpus below, select documents that should remain for "
            "high-quality LLM pretraining.\n"
            f"{budget_line}"
            'Return JSON: {"keep": ["id1", "id2", ...]}\n\n'
            + "\n".join(blocks)
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(raw, dict) and "keep" in raw:
            keep = normalize_id_list(raw.get("keep"))
            if keep is None:
                keep = []
            return {"keep": keep}
        if isinstance(raw, dict) and "remove" in raw:
            remove = normalize_id_list(raw.get("remove"))
            if remove is None:
                remove = []
            return {"remove": remove}
        keep = normalize_id_list(raw)
        if keep is None:
            raise ValueError(f"invalid keep list: {raw!r}")
        return {"keep": keep}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_corpus_filtering(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        keep = sample.label.get("keep") or sample.label.get("gold_keep") or []
        return {"keep": list(keep)}
