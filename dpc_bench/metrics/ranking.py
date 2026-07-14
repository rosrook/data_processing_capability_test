"""Ranking / correlation metrics (Level-1)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks for ties (1-based)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(_ranks(xs), _ranks(ys))


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    if total == 0:
        return None
    return (concord - discord) / total


def normalize_id_list(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("ranking", "order", "ordered_ids", "ids", "selected", "schedule"):
            if key in raw:
                return normalize_id_list(raw[key])
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(">", ",").split(",") if p.strip()]
        return parts or None
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out or None
    return None


def scores_from_ranking(order: Sequence[str], universe: Sequence[str]) -> List[float]:
    """Higher rank position (earlier in order) → higher score."""
    rank = {sid: float(len(order) - i) for i, sid in enumerate(order)}
    # unseen get 0
    return [rank.get(sid, 0.0) for sid in universe]


def evaluate_ranking(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "mean_spearman": 0.0,
            "mean_kendall": 0.0,
            "valid_rate": 0.0,
            "primary": 0.0,
            "primary_name": "mean_spearman",
            "parse_errors": 0,
        }

    spearman_vals: List[float] = []
    kendall_vals: List[float] = []
    valid = parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        gold_order = normalize_id_list(label.get("ranking") or label.get("order"))
        universe = [str(x.get("id")) for x in (inp.get("candidates") or []) if isinstance(x, dict) and x.get("id")]
        if not universe:
            universe = list(gold_order or [])
        hyp_order = normalize_id_list(pred)
        if pred is None or row.get("error") or hyp_order is None or gold_order is None:
            parse_errors += 1
            spearman_vals.append(0.0)
            continue
        # validity: permutation-ish of universe (allow subset if same length as gold)
        covers = set(hyp_order) == set(universe) or (
            len(hyp_order) == len(gold_order) and set(hyp_order).issubset(set(universe) or set(gold_order))
        )
        if not covers:
            spearman_vals.append(0.0)
            continue
        valid += 1
        gold_scores = scores_from_ranking(gold_order, universe if universe else gold_order)
        hyp_scores = scores_from_ranking(hyp_order, universe if universe else gold_order)
        sp = spearman_corr(gold_scores, hyp_scores)
        kd = kendall_tau(gold_scores, hyp_scores)
        spearman_vals.append(float(sp) if sp is not None else 0.0)
        if kd is not None:
            kendall_vals.append(float(kd))

    mean_sp = sum(spearman_vals) / n
    mean_kd = (sum(kendall_vals) / len(kendall_vals)) if kendall_vals else 0.0
    return {
        "n": n,
        "mean_spearman": round(mean_sp, 6),
        "mean_kendall": round(mean_kd, 6),
        "valid_rate": round(valid / n, 6),
        "primary": round(mean_sp, 6),
        "primary_name": "mean_spearman",
        "parse_errors": parse_errors,
    }
