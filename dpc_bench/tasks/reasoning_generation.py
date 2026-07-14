"""Task 7: Reasoning Data Generation (Level-1: solve-with-reasoning)."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.reasoning import evaluate_reasoning_generation, normalize_answer
from ..models import Sample
from .base import Task


class ReasoningGenerationTask(Task):
    """Level-1 operationalization: given a math question, generate reasoning + answer.

    Design doc Task 7 also allows free generation from a capability tag; seed rows
    primarily use GSM8K questions so answer matching is available.
    """

    name = "reasoning_generation"
    seed_filename = "reasoning_generation.jsonl"
    required_keys = ["reasoning", "answer"]
    system_prompt = (
        "You create reliable multi-step reasoning data for math word problems. "
        "Respond with JSON only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        question = sample.input.get("question") or ""
        capability = sample.input.get("capability") or "multi-step arithmetic reasoning"
        return (
            "Produce a reasoning-trace training sample for the target capability.\n"
            "Return JSON with keys:\n"
            '  "question": string (repeat or lightly rephrase the given question),\n'
            '  "reasoning": string (step-by-step),\n'
            '  "answer": string (final short answer only)\n\n'
            f"Target capability: {capability}\n"
            f"Question:\n{question}\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = raw.get("reasoning") or raw.get("solution")
        answer = raw.get("answer")
        if not isinstance(reasoning, str) or answer is None:
            raise ValueError("reasoning and answer are required")
        question = raw.get("question")
        out: Dict[str, Any] = {
            "reasoning": reasoning.strip(),
            "answer": str(answer).strip(),
        }
        if isinstance(question, str) and question.strip():
            out["question"] = question.strip()
        return out

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_reasoning_generation(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        ans = sample.label.get("answer") or sample.label.get("reference_answer") or "0"
        return {
            "question": str(sample.input.get("question") or "demo question"),
            "reasoning": (
                "Step 1: parse the quantities.\n"
                "Step 2: compute the result carefully.\n"
                f"Therefore the answer is {ans}."
            ),
            "answer": str(normalize_answer(ans) or ans),
        }
