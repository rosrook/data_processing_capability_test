"""Level-1 metrics for dataset mixing / allocation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


def _as_float_map(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, Mapping):
        return None
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            return None
    return out


def normalize_mixture(raw: Any) -> Optional[Dict[str, float]]:
    """Accept {ratios: {...}} or flat ratio map."""
    if isinstance(raw, Mapping) and "ratios" in raw:
        return _as_float_map(raw.get("ratios"))
    return _as_float_map(raw)


def score_allocation(
    prediction: Optional[Dict[str, Any]],
    *,
    domains: Iterable[str],
    reference: Mapping[str, float],
) -> Dict[str, Any]:
    domain_list = [str(d) for d in domains]
    pred = normalize_mixture(prediction)
    if pred is None:
        return {
            "valid": False,
            "covers_domains": False,
            "sum_ok": False,
            "nonneg_ok": False,
            "l1_distance": None,
            "score": 0.0,
        }

    nonneg_ok = all(v >= -1e-9 for v in pred.values())
    total = sum(pred.values())
    sum_ok = abs(total - 100.0) <= 1.5 or abs(total - 1.0) <= 0.015
    # Normalize to percentages for distance
    scale = 100.0 if total <= 1.5 else 1.0
    pred_pct = {k: v * scale for k, v in pred.items()} if total <= 1.5 else dict(pred)

    covers = all(d in pred_pct for d in domain_list)
    # Ignore unknown extra keys but penalize if required missing
    l1 = 0.0
    for d in domain_list:
        ref = float(reference.get(d, 0.0))
        hyp = float(pred_pct.get(d, 0.0))
        l1 += abs(ref - hyp)
    for k, v in pred_pct.items():
        if k not in reference and k not in domain_list:
            l1 += abs(float(v))

    # score: validity + distance quality (lower L1 better; max L1 ~200)
    valid = nonneg_ok and sum_ok and covers
    distance_score = max(0.0, 1.0 - (l1 / 200.0)) if valid else 0.0
    score = (0.4 if valid else 0.0) + 0.6 * distance_score

    return {
        "valid": valid,
        "covers_domains": covers,
        "sum_ok": sum_ok,
        "nonneg_ok": nonneg_ok,
        "l1_distance": round(l1, 6),
        "score": round(score, 6),
    }


def evaluate_allocation(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "valid_rate": 0.0,
            "mean_l1": None,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_score",
            "parse_errors": 0,
        }

    valid = parse_errors = 0
    scores = []
    l1s = []
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        meta = row.get("meta") or {}
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        if pred is None or row.get("error"):
            parse_errors += 1
            scores.append(0.0)
            continue
        reference = label.get("reference") or {}
        domains = (
            label.get("domains")
            or meta.get("domains")
            or inp.get("domains")
            or list(reference.keys())
        )
        detail = score_allocation(pred, domains=domains, reference=reference)
        scores.append(float(detail["score"]))
        if detail["valid"]:
            valid += 1
            if detail["l1_distance"] is not None:
                l1s.append(float(detail["l1_distance"]))
        else:
            if detail["l1_distance"] is None:
                pass

    mean_score = sum(scores) / n
    mean_l1 = (sum(l1s) / len(l1s)) if l1s else None
    return {
        "n": n,
        "valid_rate": round(valid / n, 6),
        "mean_l1": round(mean_l1, 6) if mean_l1 is not None else None,
        "mean_score": round(mean_score, 6),
        "primary": round(mean_score, 6),
        "primary_name": "mean_score",
        "parse_errors": parse_errors,
    }
