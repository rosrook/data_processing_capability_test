"""Curriculum scheduling Level-1 metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .ranking import normalize_id_list, spearman_corr, scores_from_ranking


_PHASE_ORDER = {"easy": 0, "medium": 1, "hard": 2, "easy_first": 0}


def normalize_phases(raw: Any) -> Optional[List[str]]:
    order = normalize_id_list(raw)
    if order is None:
        return None
    out = []
    for p in order:
        key = p.strip().lower().replace(" ", "_")
        if key in {"easy", "medium", "hard"}:
            out.append(key)
        else:
            out.append(key)
    return out or None


def score_curriculum(
    prediction: Any,
    *,
    gold_schedule: Sequence[str],
    mode: str = "phases",
) -> Dict[str, Any]:
    if mode == "phases":
        hyp = normalize_phases(prediction)
        gold = [str(x).lower() for x in gold_schedule]
        if hyp is None:
            return {"valid": False, "order_score": 0.0, "score": 0.0}
        # valid if permutation of easy/medium/hard or exact gold length
        valid = len(hyp) == len(gold) and set(hyp) == set(gold)
        # score by phase index correlation
        gold_scores = [_PHASE_ORDER.get(p, i) for i, p in enumerate(gold)]
        # predicted: position as training order (earlier = lower difficulty preferred)
        hyp_pos = {p: i for i, p in enumerate(hyp)}
        hyp_scores = [float(hyp_pos.get(p, len(hyp))) for p in gold]
        # ideal: increasing difficulty → gold positions 0,1,2 vs hyp positions
        # Use kendall between gold difficulty rank and hyp training position
        ideal = list(range(len(gold)))
        # Spearman between predicted position and ideal easy→hard positions
        sp = spearman_corr(ideal, hyp_scores)
        # invert: hyp_scores are positions; ideal easy first means positions 0,1,2 for easy,med,hard
        # Better: compare hyp phase sequence directly to gold
        exact = 1.0 if hyp == gold else 0.0
        prefix = 0.0
        for a, b in zip(hyp, gold):
            if a == b:
                prefix += 1.0
            else:
                break
        prefix /= max(1, len(gold))
        order_score = 0.5 * exact + 0.5 * prefix if valid else 0.2 * prefix
        if sp is not None and valid:
            # high correlation of positions with ideal means good
            order_score = max(order_score, 0.3 + 0.7 * max(0.0, (sp + 1) / 2))
        score = (0.3 if valid else 0.0) + 0.7 * order_score
        return {"valid": valid, "order_score": round(order_score, 6), "score": round(min(1.0, score), 6)}

    # item-id schedule
    hyp = normalize_id_list(prediction)
    gold = [str(x) for x in gold_schedule]
    if hyp is None or not gold:
        return {"valid": False, "order_score": 0.0, "score": 0.0}
    valid = set(hyp) == set(gold) and len(hyp) == len(gold)
    g_scores = scores_from_ranking(gold, gold)
    h_scores = scores_from_ranking(hyp, gold)
    sp = spearman_corr(g_scores, h_scores)
    order_score = float(sp) if sp is not None else 0.0
    # map spearman [-1,1] → [0,1]
    order_score = (order_score + 1.0) / 2.0
    score = (0.3 if valid else 0.0) + 0.7 * order_score
    return {"valid": valid, "order_score": round(order_score, 6), "score": round(score, 6)}


def evaluate_curriculum(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "valid_rate": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }
    scores: List[float] = []
    valid = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        gold = label.get("schedule") or label.get("order") or []
        mode = str(label.get("mode") or "phases")
        if pred is None or row.get("error"):
            parse_errors += 1
            scores.append(0.0)
            continue
        detail = score_curriculum(pred, gold_schedule=gold, mode=mode)
        scores.append(float(detail["score"]))
        valid += int(detail["valid"])
    mean_score = sum(scores) / n
    return {
        "n": n,
        "valid_rate": round(valid / n, 6),
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }
