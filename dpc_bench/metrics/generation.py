"""Level-1 metrics for instruction generation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _nonempty_str(value: Any, *, min_len: int = 8) -> bool:
    return isinstance(value, str) and len(value.strip()) >= min_len


def score_instruction_sample(
    prediction: Optional[Dict[str, Any]],
    *,
    topic: str = "",
    requires_reasoning: bool = False,
) -> Dict[str, Any]:
    if not isinstance(prediction, dict):
        return {
            "valid_json": False,
            "schema_ok": False,
            "fields_nonempty": False,
            "heuristic_ok": False,
            "score": 0.0,
        }

    instruction = prediction.get("instruction")
    inp = prediction.get("input", "")
    output = prediction.get("output")

    schema_ok = (
        isinstance(instruction, str)
        and isinstance(output, str)
        and (inp is None or isinstance(inp, str))
    )
    fields_nonempty = _nonempty_str(instruction, min_len=12) and _nonempty_str(output, min_len=12)
    if isinstance(inp, str) and inp.strip():
        fields_nonempty = fields_nonempty and len(inp.strip()) >= 1

    heuristic_ok = fields_nonempty
    instr_l = (instruction or "").lower() if isinstance(instruction, str) else ""
    out_l = (output or "").lower() if isinstance(output, str) else ""

    # Reject obvious placeholder / self-reference junk
    bad_markers = ("todo", "tbd", "as an ai", "i cannot", "lorem ipsum", "placeholder")
    if any(m in instr_l or m in out_l for m in bad_markers):
        heuristic_ok = False
    if instruction and output and instruction.strip() == output.strip():
        heuristic_ok = False
    if requires_reasoning:
        reason_markers = ("because", "step", "first", "therefore", "thus", "reason", "所以", "首先", "因此")
        if not any(m in out_l for m in reason_markers):
            heuristic_ok = False
    if topic:
        # Soft signal: topic word appearing somewhere is good but not required
        pass

    score = 0.0
    if schema_ok:
        score += 0.35
    if fields_nonempty:
        score += 0.35
    if heuristic_ok:
        score += 0.30

    return {
        "valid_json": True,
        "schema_ok": schema_ok,
        "fields_nonempty": fields_nonempty,
        "heuristic_ok": heuristic_ok,
        "score": round(score, 6),
    }


def evaluate_instruction_generation(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "valid_rate": 0.0,
            "schema_rate": 0.0,
            "nonempty_rate": 0.0,
            "heuristic_rate": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }

    scores: List[float] = []
    valid = schema = nonempty = heuristic = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        if pred is None or row.get("error"):
            parse_errors += 1
            scores.append(0.0)
            continue
        meta = row.get("meta") or {}
        detail = score_instruction_sample(
            pred,
            topic=str((row.get("label") or {}).get("topic") or meta.get("topic") or ""),
            requires_reasoning=bool(meta.get("requires_reasoning")),
        )
        scores.append(float(detail["score"]))
        valid += int(detail["valid_json"])
        schema += int(detail["schema_ok"])
        nonempty += int(detail["fields_nonempty"])
        heuristic += int(detail["heuristic_ok"])

    mean_score = sum(scores) / n
    return {
        "n": n,
        "valid_rate": round(valid / n, 6),
        "schema_rate": round(schema / n, 6),
        "nonempty_rate": round(nonempty / n, 6),
        "heuristic_rate": round(heuristic / n, 6),
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }
