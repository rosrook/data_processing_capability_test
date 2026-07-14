"""Reasoning / hard-sample generation Level-1 metrics."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    # GSM8K style #### answer
    if "####" in s:
        s = s.split("####")[-1].strip()
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        return m.group(0).lstrip("+")
    return s.lower()


def extract_answer_field(prediction: Optional[Dict[str, Any]]) -> str:
    if not isinstance(prediction, dict):
        return ""
    for key in ("answer", "final_answer", "result"):
        if prediction.get(key) is not None:
            return normalize_answer(prediction.get(key))
    # fall back: last number in solution/reasoning
    blob = " ".join(
        str(prediction.get(k) or "") for k in ("solution", "reasoning", "output")
    )
    return normalize_answer(blob)


def score_reasoning_sample(
    prediction: Optional[Dict[str, Any]],
    *,
    gold_answer: str = "",
) -> Dict[str, Any]:
    if not isinstance(prediction, dict):
        return {
            "schema_ok": False,
            "has_reasoning": False,
            "answer_match": False,
            "score": 0.0,
        }

    question = prediction.get("question")
    reasoning = prediction.get("reasoning") or prediction.get("solution")
    answer = prediction.get("answer")
    # When solving a provided question, question field may be omitted
    schema_ok = isinstance(reasoning, str) and (
        isinstance(answer, str) or isinstance(answer, (int, float)) or answer is not None
    )
    if question is not None and not isinstance(question, str):
        schema_ok = False

    reason_text = str(reasoning or "")
    has_reasoning = len(reason_text.strip()) >= 40 and any(
        m in reason_text.lower()
        for m in ("step", "first", "because", "so ", "therefore", "thus", "####", "所以", "首先")
    )
    hyp = extract_answer_field(prediction)
    gold = normalize_answer(gold_answer)
    answer_match = bool(gold) and hyp == gold

    score = 0.0
    if schema_ok:
        score += 0.35
    if has_reasoning:
        score += 0.25
    if answer_match:
        score += 0.40
    elif hyp and gold:
        score += 0.05  # attempted numeric answer

    return {
        "schema_ok": schema_ok,
        "has_reasoning": has_reasoning,
        "answer_match": answer_match,
        "score": round(score, 6),
    }


def evaluate_reasoning_generation(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "schema_rate": 0.0,
            "answer_accuracy": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }

    scores: List[float] = []
    schema = answers = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        gold = label.get("answer") or label.get("reference_answer") or ""
        if pred is None or row.get("error"):
            parse_errors += 1
            scores.append(0.0)
            continue
        detail = score_reasoning_sample(pred, gold_answer=str(gold))
        scores.append(float(detail["score"]))
        schema += int(detail["schema_ok"])
        answers += int(detail["answer_match"])

    mean_score = sum(scores) / n
    return {
        "n": n,
        "schema_rate": round(schema / n, 6),
        "answer_accuracy": round(answers / n, 6),
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }


def score_hard_sample(prediction: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(prediction, dict):
        return {"schema_ok": False, "hard_markers": False, "score": 0.0}
    q = prediction.get("question") or prediction.get("sample")
    a = prediction.get("answer")
    why = prediction.get("why_hard") or prediction.get("rationale")
    schema_ok = isinstance(q, str) and len(q.strip()) >= 20 and a is not None
    text = f"{q}\n{why or ''}".lower()
    markers = (
        "multi-step",
        "trap",
        "distractor",
        "ambiguous",
        "edge case",
        "hard",
        "misleading",
        "多步",
        "陷阱",
    )
    hard_markers = any(m in text for m in markers) or (isinstance(q, str) and len(q) >= 120)
    score = 0.4 * float(schema_ok) + 0.4 * float(hard_markers)
    if isinstance(why, str) and len(why.strip()) >= 20:
        score += 0.2
    return {
        "schema_ok": schema_ok,
        "hard_markers": hard_markers,
        "score": round(min(1.0, score), 6),
    }


def evaluate_hard_sample_generation(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "schema_rate": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }
    scores: List[float] = []
    schema = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        if pred is None or row.get("error"):
            parse_errors += 1
            scores.append(0.0)
            continue
        detail = score_hard_sample(pred)
        scores.append(float(detail["score"]))
        schema += int(detail["schema_ok"])
    mean_score = sum(scores) / n
    return {
        "n": n,
        "schema_rate": round(schema / n, 6),
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }
