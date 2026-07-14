"""Format repair Level-1 metrics."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def _as_object(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        # unwrap common wrappers
        if "repaired" in value and isinstance(value["repaired"], dict):
            return value["repaired"]
        if "json" in value and isinstance(value["json"], dict):
            return value["json"]
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def score_format_repair(
    prediction: Any,
    *,
    target: Dict[str, Any],
    required_keys: Iterable[str],
) -> Dict[str, Any]:
    keys = [str(k) for k in required_keys]
    pred = _as_object(prediction)
    if pred is None:
        return {
            "valid_json": False,
            "schema_ok": False,
            "key_recall": 0.0,
            "value_match": 0.0,
            "score": 0.0,
        }

    schema_ok = all(k in pred for k in keys)
    key_recall = (sum(1 for k in keys if k in pred) / len(keys)) if keys else 1.0
    matches = 0
    comparable = 0
    for k in keys:
        if k not in target:
            continue
        comparable += 1
        if k in pred and str(pred.get(k)).strip() == str(target.get(k)).strip():
            matches += 1
    value_match = (matches / comparable) if comparable else 0.0
    # valid_json implied when pred parsed as object
    score = 0.25 + 0.25 * float(schema_ok) + 0.25 * key_recall + 0.25 * value_match
    return {
        "valid_json": True,
        "schema_ok": schema_ok,
        "key_recall": round(key_recall, 6),
        "value_match": round(value_match, 6),
        "score": round(score, 6),
    }


def evaluate_format_repair(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "valid_rate": 0.0,
            "schema_rate": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }

    scores: List[float] = []
    valid = schema = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        target = label.get("target") or label.get("repaired") or {}
        required = label.get("required_keys") or list(target.keys()) or ["instruction", "output"]
        if pred is None or row.get("error") or not isinstance(target, dict):
            parse_errors += 1
            scores.append(0.0)
            continue
        detail = score_format_repair(pred, target=target, required_keys=required)
        scores.append(float(detail["score"]))
        valid += int(detail["valid_json"])
        schema += int(detail["schema_ok"])

    mean_score = sum(scores) / n
    return {
        "n": n,
        "valid_rate": round(valid / n, 6),
        "schema_rate": round(schema / n, 6),
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }
