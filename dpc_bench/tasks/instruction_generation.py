"""Task 6 (Phase 1): Instruction Generation."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.generation import evaluate_instruction_generation
from ..models import Sample
from .base import Task


class InstructionGenerationTask(Task):
    name = "instruction_generation"
    seed_filename = "instruction_generation.jsonl"
    required_keys = ["instruction", "output"]
    system_prompt = (
        "You generate high-quality instruction-tuning samples. "
        "Respond with a single JSON object containing instruction, input, and output."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        topic = sample.input.get("topic", "")
        constraints = sample.input.get("constraints") or {}
        difficulty = constraints.get("difficulty", "medium")
        requires_reasoning = constraints.get("requires_reasoning", False)
        return (
            "Generate one instruction-tuning sample as JSON with keys:\n"
            '  "instruction": string,\n'
            '  "input": string (may be empty),\n'
            '  "output": string\n\n'
            f"Topic: {topic}\n"
            f"Constraints:\n"
            f"- difficulty: {difficulty}\n"
            f"- requires_reasoning: {requires_reasoning}\n"
            "The output must answer the instruction correctly and be useful for SFT.\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        instruction = raw.get("instruction")
        output = raw.get("output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise ValueError("instruction and output must be strings")
        inp = raw.get("input", "")
        if inp is None:
            inp = ""
        if not isinstance(inp, str):
            raise ValueError("input must be a string")
        return {"instruction": instruction.strip(), "input": inp.strip(), "output": output.strip()}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        enriched = []
        for row in records:
            meta = dict(row.get("meta") or {})
            inp = row.get("input") if isinstance(row.get("input"), dict) else {}
            label = row.get("label") or {}
            constraints = label.get("constraints") or inp.get("constraints") or {}
            if isinstance(constraints, dict):
                meta.setdefault("requires_reasoning", bool(constraints.get("requires_reasoning")))
            meta.setdefault("topic", inp.get("topic") or label.get("topic") or "")
            enriched.append({**row, "meta": meta})
        return evaluate_instruction_generation(enriched)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        topic = sample.input.get("topic", "general knowledge")
        requires = bool((sample.input.get("constraints") or {}).get("requires_reasoning"))
        if requires:
            output = (
                f"Step 1: identify the key concepts about {topic}. "
                f"Step 2: reason about how they relate. "
                f"Therefore, a concise correct answer follows."
            )
        else:
            output = f"Here is a clear, useful explanation about {topic} with concrete details."
        return {
            "instruction": f"Explain a practical concept related to {topic} for a learner.",
            "input": "",
            "output": output,
        }
