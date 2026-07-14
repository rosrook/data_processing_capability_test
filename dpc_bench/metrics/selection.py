"""Diversity selection Level-1 metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .ranking import normalize_id_list


def _cluster_of(candidates: Sequence[Dict[str, Any]], sid: str) -> str:
    for c in candidates:
        if str(c.get("id")) == sid:
            return str(c.get("cluster") or c.get("topic") or "unknown")
    return "unknown"


def score_diversity_selection(
    prediction: Any,
    *,
    candidates: Sequence[Dict[str, Any]],
    budget: int,
    reference: Sequence[str],
) -> Dict[str, Any]:
    hyp = normalize_id_list(prediction)
    universe = {str(c.get("id")) for c in candidates if c.get("id")}
    if hyp is None:
        return {
            "valid": False,
            "budget_ok": False,
            "coverage": 0.0,
            "jaccard": 0.0,
            "score": 0.0,
        }

    # unique, in-universe
    seen: List[str] = []
    for d in hyp:
        if d in universe and d not in seen:
            seen.append(d)
    budget_ok = len(seen) == int(budget)
    valid = budget_ok and len(seen) > 0

    clusters = {_cluster_of(candidates, sid) for sid in seen}
    all_clusters = {_cluster_of(candidates, str(c["id"])) for c in candidates if c.get("id")}
    coverage = (len(clusters) / len(all_clusters)) if all_clusters else 0.0

    ref_set: Set[str] = set(str(x) for x in reference)
    hyp_set = set(seen)
    inter = len(ref_set & hyp_set)
    union = len(ref_set | hyp_set) or 1
    jaccard = inter / union

    score = (0.3 if valid else 0.0) + 0.4 * coverage + 0.3 * jaccard
    return {
        "valid": valid,
        "budget_ok": budget_ok,
        "coverage": round(coverage, 6),
        "jaccard": round(jaccard, 6),
        "score": round(score, 6),
    }


def evaluate_diversity_selection(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "valid_rate": 0.0,
            "mean_coverage": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }

    scores: List[float] = []
    coverages: List[float] = []
    valid = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        candidates = inp.get("candidates") or []
        budget = int(inp.get("budget") or label.get("budget") or 0)
        reference = normalize_id_list(label.get("selected") or label.get("reference")) or []
        if pred is None or row.get("error"):
            parse_errors += 1
            scores.append(0.0)
            continue
        detail = score_diversity_selection(
            pred, candidates=candidates, budget=budget, reference=reference
        )
        scores.append(float(detail["score"]))
        coverages.append(float(detail["coverage"]))
        valid += int(detail["valid"])

    mean_score = sum(scores) / n
    return {
        "n": n,
        "valid_rate": round(valid / n, 6),
        "mean_coverage": round(sum(coverages) / n, 6) if coverages else 0.0,
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }
