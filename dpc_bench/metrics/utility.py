"""Data utility prediction Level-1 metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .ranking import spearman_corr


def _as_score(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        for key in ("utility", "utility_score", "score", "value"):
            if key in raw:
                return _as_score(raw.get(key))
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def pairwise_accuracy(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    n = len(y_true)
    if n != len(y_pred) or n < 2:
        return None
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            if dt == 0:
                continue
            total += 1
            if dt * dp > 0:
                correct += 1
            elif dp == 0:
                pass
    if total == 0:
        return None
    return correct / total


def evaluate_utility_prediction(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "spearman": 0.0,
            "pairwise_accuracy": 0.0,
            "valid_rate": 0.0,
            "primary": 0.0,
            "primary_name": "spearman",
            "parse_errors": 0,
        }

    y_true: List[float] = []
    y_pred: List[float] = []
    valid = parse_errors = 0
    for row in rows:
        label = row.get("label") or {}
        gold = _as_score(label.get("utility") if "utility" in label else label.get("score"))
        hyp = _as_score(row.get("prediction"))
        if row.get("error") or gold is None or hyp is None:
            parse_errors += 1
            continue
        valid += 1
        y_true.append(float(gold))
        y_pred.append(float(hyp))

    sp = spearman_corr(y_true, y_pred) if len(y_true) >= 2 else None
    pw = pairwise_accuracy(y_true, y_pred) if len(y_true) >= 2 else None
    # primary in [0,1]-ish: map spearman from [-1,1]
    primary = ((sp + 1.0) / 2.0) if sp is not None else 0.0
    return {
        "n": n,
        "spearman": round(sp, 6) if sp is not None else None,
        "pairwise_accuracy": round(pw, 6) if pw is not None else None,
        "valid_rate": round(valid / n, 6),
        "primary": round(primary, 6),
        "primary_name": "spearman_norm",
        "parse_errors": parse_errors,
    }
