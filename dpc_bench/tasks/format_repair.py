"""Task 3: Format Repair."""

from __future__ import annotations

from typing import Any, Dict, List

from ..metrics.repair import evaluate_format_repair
from ..models import Sample
from .base import Task


class FormatRepairTask(Task):
    name = "format_repair"
    seed_filename = "format_repair.jsonl"
    required_keys = ["instruction", "output"]
    system_prompt = (
        "You repair malformed instruction-tuning JSON records. "
        "Respond with a single valid JSON object only."
    )

    def build_user_prompt(self, sample: Sample) -> str:
        broken = sample.input.get("broken_json", "")
        schema = sample.input.get("schema_hint") or ["instruction", "input", "output"]
        return (
            "Repair the following malformed JSON into a valid object with keys "
            f"{schema}. Preserve meaning when possible; invent minimal placeholders "
            "only for missing required fields.\n"
            'Return JSON: {"instruction": str, "input": str, "output": str}\n\n'
            f"Broken JSON:\n{broken}\n"
        )

    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        instr = raw.get("instruction")
        output = raw.get("output")
        if not isinstance(instr, str) or not isinstance(output, str):
            raise ValueError("instruction and output must be strings")
        inp = raw.get("input", "")
        if inp is None:
            inp = ""
        if not isinstance(inp, str):
            raise ValueError("input must be a string")
        return {"instruction": instr.strip(), "input": inp.strip(), "output": output.strip()}

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate_format_repair(records)

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        target = sample.label.get("target") or sample.label.get("repaired") or {}
        return {
            "instruction": str(target.get("instruction") or "placeholder instruction"),
            "input": str(target.get("input") or ""),
            "output": str(target.get("output") or "placeholder output"),
        }
