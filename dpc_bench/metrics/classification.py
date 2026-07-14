"""Binary classification metrics."""

from __future__ import annotations

from typing import Iterable, List, Optional


def binary_classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[Optional[str]],
    *,
    positive: str,
) -> dict:
    truths = list(y_true)
    preds = list(y_pred)
    if len(truths) != len(preds):
        raise ValueError("y_true and y_pred length mismatch")

    tp = fp = tn = fn = 0
    parse_errors = 0
    correct = 0
    for t, p in zip(truths, preds):
        if p is None:
            parse_errors += 1
            # Missing prediction counts as incorrect; for F1, treat as false negative
            # when gold is positive, else ignore for confusion (still hurts accuracy).
            if t == positive:
                fn += 1
            continue
        if p == t:
            correct += 1
        if p == positive and t == positive:
            tp += 1
        elif p == positive and t != positive:
            fp += 1
        elif p != positive and t != positive:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = len(truths)
    accuracy = correct / n if n else 0.0

    return {
        "n": n,
        "positive_label": positive,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "parse_errors": parse_errors,
        "primary": round(f1, 6),
        "primary_name": "f1",
    }


def normalize_yes_no(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in {"YES", "Y", "TRUE", "1", "DUPLICATE"}:
        return "YES"
    if s in {"NO", "N", "FALSE", "0", "NOT_DUPLICATE", "DIFFERENT"}:
        return "NO"
    return None


def normalize_keep_remove(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"keep", "yes", "high", "good", "accept", "1", "true"}:
        return "keep"
    if s in {"remove", "no", "low", "bad", "reject", "0", "false"}:
        return "remove"
    return None
