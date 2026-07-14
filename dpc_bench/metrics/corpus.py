"""Corpus-level filtering / dedup Level-1 metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .ranking import normalize_id_list


def _as_id_set(raw: Any, *, key_candidates: Sequence[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in key_candidates:
            if key in raw:
                ids = normalize_id_list(raw.get(key))
                return set(ids) if ids is not None else set()
        # decisions map: {id: keep/remove}
        decisions = raw.get("decisions")
        if isinstance(decisions, dict):
            keep = set()
            for sid, val in decisions.items():
                s = str(val).strip().lower()
                if s in {"keep", "yes", "true", "1", "retain"}:
                    keep.add(str(sid))
            return keep
        return None
    ids = normalize_id_list(raw)
    return set(ids) if ids is not None else None


def set_prf(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    if not gold and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def score_corpus_filtering(
    prediction: Any,
    *,
    universe: Sequence[str],
    gold_keep: Sequence[str],
) -> Dict[str, Any]:
    uni = {str(x) for x in universe}
    gold = {str(x) for x in gold_keep} & uni
    pred_keep = _as_id_set(prediction, key_candidates=("keep", "selected", "retain", "ids"))
    if pred_keep is None:
        # allow remove-list form
        pred_remove = _as_id_set(prediction, key_candidates=("remove", "discard", "reject"))
        if pred_remove is None:
            return {
                "valid": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "score": 0.0,
            }
        pred_keep = uni - pred_remove
    else:
        pred_keep = pred_keep & uni

    valid = True
    prf = set_prf(pred_keep, gold)
    # Primary for dirty-corpus filtering: recall of true keep docs, with F1 as score blend
    score = 0.55 * prf["recall"] + 0.45 * prf["f1"]
    return {
        "valid": valid,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "score": round(score, 6),
        "n_pred_keep": len(pred_keep),
        "n_gold_keep": len(gold),
    }


def evaluate_corpus_filtering(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "mean_recall": 0.0,
            "mean_precision": 0.0,
            "mean_f1": 0.0,
            "primary": 0.0,
            "primary_name": "mean_recall",
            "parse_errors": 0,
        }

    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        docs = inp.get("documents") or []
        universe = [str(d.get("id")) for d in docs if isinstance(d, dict) and d.get("id")]
        gold_keep = label.get("keep") or label.get("gold_keep") or []
        if pred is None or row.get("error"):
            parse_errors += 1
            recalls.append(0.0)
            precisions.append(0.0)
            f1s.append(0.0)
            continue
        detail = score_corpus_filtering(pred, universe=universe, gold_keep=gold_keep)
        recalls.append(float(detail["recall"]))
        precisions.append(float(detail["precision"]))
        f1s.append(float(detail["f1"]))

    return {
        "n": n,
        "mean_recall": round(sum(recalls) / n, 6),
        "mean_precision": round(sum(precisions) / n, 6),
        "mean_f1": round(sum(f1s) / n, 6),
        "primary": round(sum(recalls) / n, 6),
        "primary_name": "mean_recall",
        "parse_errors": parse_errors,
    }


def _gold_pairs(clusters: Sequence[Sequence[str]]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for cluster in clusters:
        ids = [str(x) for x in cluster]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                pairs.add((a, b))
    return pairs


def score_corpus_dedup(
    prediction: Any,
    *,
    universe: Sequence[str],
    clusters: Sequence[Sequence[str]],
    gold_remove: Sequence[str],
) -> Dict[str, Any]:
    uni = {str(x) for x in universe}
    gold_rm = {str(x) for x in gold_remove} & uni
    pred_rm = _as_id_set(prediction, key_candidates=("remove", "discard", "duplicates"))
    if pred_rm is None:
        pred_keep = _as_id_set(prediction, key_candidates=("keep", "survivors", "retain"))
        if pred_keep is None:
            return {
                "valid": False,
                "remove_recall": 0.0,
                "remove_precision": 0.0,
                "remove_f1": 0.0,
                "pair_recall": 0.0,
                "score": 0.0,
            }
        pred_rm = uni - (pred_keep & uni)
    else:
        pred_rm = pred_rm & uni

    prf = set_prf(pred_rm, gold_rm)
    pairs = _gold_pairs(clusters)
    if pairs:
        hit = 0
        for a, b in pairs:
            # pair resolved if not both kept
            kept = uni - pred_rm
            if not (a in kept and b in kept):
                hit += 1
        pair_recall = hit / len(pairs)
    else:
        pair_recall = 1.0

    score = 0.4 * prf["recall"] + 0.2 * prf["precision"] + 0.4 * pair_recall
    return {
        "valid": True,
        "remove_recall": prf["recall"],
        "remove_precision": prf["precision"],
        "remove_f1": prf["f1"],
        "pair_recall": round(pair_recall, 6),
        "score": round(score, 6),
    }


def evaluate_corpus_dedup(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "mean_remove_recall": 0.0,
            "mean_pair_recall": 0.0,
            "mean_score": 0.0,
            "primary": 0.0,
            "primary_name": "mean_pair_recall",
            "parse_errors": 0,
        }

    rm_recalls: List[float] = []
    pair_recalls: List[float] = []
    scores: List[float] = []
    parse_errors = 0
    for row in rows:
        pred = row.get("prediction")
        label = row.get("label") or {}
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        docs = inp.get("documents") or []
        universe = [str(d.get("id")) for d in docs if isinstance(d, dict) and d.get("id")]
        clusters = label.get("clusters") or []
        gold_remove = label.get("remove") or []
        if pred is None or row.get("error"):
            parse_errors += 1
            rm_recalls.append(0.0)
            pair_recalls.append(0.0)
            scores.append(0.0)
            continue
        detail = score_corpus_dedup(
            pred, universe=universe, clusters=clusters, gold_remove=gold_remove
        )
        rm_recalls.append(float(detail["remove_recall"]))
        pair_recalls.append(float(detail["pair_recall"]))
        scores.append(float(detail["score"]))

    return {
        "n": n,
        "mean_remove_recall": round(sum(rm_recalls) / n, 6),
        "mean_pair_recall": round(sum(pair_recalls) / n, 6),
        "mean_score": round(sum(scores) / n, 6),
        "primary": round(sum(pair_recalls) / n, 6),
        "primary_name": "mean_pair_recall",
        "parse_errors": parse_errors,
    }
