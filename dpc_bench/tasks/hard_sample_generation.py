"""Task 8: Hard Sample Generation."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.reasoning import evaluate_hard_sample_generation
from ..models import Sample
from .base import Task


class HardSampleGenerationTask(Task):
    name = "hard_sample_generation"
    seed_filename = "hard_sample_generation.jsonl"
    required_keys = ["question", "answer"]
    system_prompt = (
        "You construct hard evaluation/training samples that expose model weaknesses. "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        capability = sample.input.get("capability") or "fraction reasoning"
        hint = sample.input.get("hint") or ""
        exemplar = sample.input.get("exemplar_problem") or ""
        exemplar_level = sample.input.get("exemplar_level") or ""
        exemplar_block = ""
        if exemplar:
            exemplar_block = (
                f"\nExemplar (style/difficulty reference"
                f"{f', {exemplar_level}' if exemplar_level else ''}):\n{exemplar}\n"
            )
        return (
            "Generate ONE new hard sample for the capability below "
            "(do not copy the exemplar verbatim).\n"
            "Return JSON:\n"
            '  {"question": str, "answer": str, "why_hard": str}\n'
            "The question should include traps, multi-step requirements, or edge cases.\n\n"
            f"Capability: {capability}\n"
            f"Hint: {hint}\n"
            f"{exemplar_block}"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        q = raw.get("question") or raw.get("sample")
        a = raw.get("answer")
        if not isinstance(q, str) or a is None:
            raise ValueError("question and answer are required")
        out: Dict[str, Any] = {"question": q.strip(), "answer": str(a).strip()}
        why = raw.get("why_hard") or raw.get("rationale")
        if isinstance(why, str):
            out["why_hard"] = why.strip()
        return out

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_hard_sample_generation(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        cap = sample.input.get("capability") or "reasoning"
        ref = sample.label.get("reference") or {}
        if ref.get("question"):
            return {
                "question": str(ref["question"]),
                "answer": str(ref.get("answer") or "0"),
                "why_hard": str(ref.get("why_hard") or "multi-step trap with distractors"),
            }
        return {
            "question": (
                f"Hard multi-step problem about {cap} with a misleading distractor: "
                "A recipe needs 3/4 cup sugar and you only have a 1/8-cup scoop; "
                "after using 2 scoops you spill 1/3 of what you scooped. "
                "How many scoops remain to finish?"
            ),
            "answer": "4",
            "why_hard": "multi-step fraction trap with distractor quantities",
        }
