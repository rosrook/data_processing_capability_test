#!/usr/bin/env python3
"""Download FULL public HF dataset files via mirror, then materialize seed JSONL.

By default this pulls all relevant splits/shards (not a 400-sample subset).
Evaluation can still subsample with:

  dpc-bench run --limit 50 ...

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/prepare_seed_from_hf.py
  python scripts/prepare_seed_from_hf.py --limit-per-task 400   # optional smaller seed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "data" / "seed"
RAW_DIR = ROOT / "data" / "hf_raw"
BACKUP_DIR = ROOT / "data" / "seed_handwritten_backup"
DEFAULT_ENDPOINT = (
    os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/") or "https://hf-mirror.com"
)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 50000 == 0:
                print(f"  ... wrote {n} rows so far -> {path.name}")
    print(f"wrote {n} -> {path}")
    return n


def _backup_existing_seed() -> None:
    if not SEED_DIR.is_dir():
        return
    files = list(SEED_DIR.glob("*.jsonl"))
    if not files:
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = BACKUP_DIR / src.name
        # refresh backup only if missing, keep first handwritten snapshot
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            print(f"backed up {src.name} -> {dst}")


def _resolve_url(repo: str, path: str, *, endpoint: str, revision: str = "main") -> str:
    quoted_path = "/".join(quote(p, safe="") for p in path.split("/"))
    return f"{endpoint}/datasets/{repo}/resolve/{revision}/{quoted_path}"


def list_repo_files(repo: str, *, endpoint: str, prefix: str = "") -> List[Dict[str, Any]]:
    import requests

    url = f"{endpoint}/api/datasets/{repo}/tree/main"
    if prefix:
        url = f"{url}/{prefix.strip('/')}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return list(resp.json())


def list_parquet_under(repo: str, *, endpoint: str, prefix: str) -> List[str]:
    entries = list_repo_files(repo, endpoint=endpoint, prefix=prefix)
    out: List[str] = []
    for e in entries:
        path = str(e.get("path") or "")
        typ = e.get("type")
        if typ == "file" and path.endswith(".parquet"):
            out.append(path)
        elif typ == "directory":
            out.extend(list_parquet_under(repo, endpoint=endpoint, prefix=path))
    return sorted(out)


def download_file(repo: str, path: str, dest: Path, *, endpoint: str, timeout: int = 600) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"skip existing {dest} ({dest.stat().st_size} bytes)")
        return dest

    urls = [
        _resolve_url(repo, path, endpoint=endpoint),
        _resolve_url(repo, path, endpoint="https://huggingface.co"),
    ]
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            print(f"downloading {url}")
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
                resp.raise_for_status()
                tmp = dest.parent / (dest.name + ".part")
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            print(f"saved {dest} ({dest.stat().st_size} bytes)")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"download failed via {url}: {exc}")
    raise RuntimeError(f"failed to download {repo}:{path}: {last_err}")


def download_repo_files(
    repo: str,
    remotes: Sequence[str],
    dest_dir: Path,
    *,
    endpoint: str,
) -> List[Path]:
    paths: List[Path] = []
    for remote in remotes:
        local = dest_dir / remote
        paths.append(download_file(repo, remote, local, endpoint=endpoint))
    return paths


def iter_parquet_rows(paths: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        table = pq.read_table(path)
        cols = table.column_names
        cols_data = {c: table.column(c).to_pylist() for c in cols}
        n = table.num_rows
        print(f"  reading {path} rows={n}")
        for i in range(n):
            yield {c: cols_data[c][i] for c in cols}


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _maybe_limit(rows: List[Dict[str, Any]], limit: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    # try preserve label balance when possible
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        lab = r.get("label") or {}
        key = str(lab.get("duplicate") or lab.get("decision") or "all")
        by_label.setdefault(key, []).append(r)
    if len(by_label) >= 2:
        keys = sorted(by_label)
        per = max(1, limit // len(keys))
        selected: List[Dict[str, Any]] = []
        for k in keys:
            pool = by_label[k]
            rng.shuffle(pool)
            selected.extend(pool[:per])
        rng.shuffle(selected)
        return selected[:limit]
    rng.shuffle(rows)
    return rows[:limit]


def build_semantic_dedup(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    paws_files = list_parquet_under(
        "google-research-datasets/paws", endpoint=endpoint, prefix="labeled_final"
    )
    if not paws_files:
        raise RuntimeError("no PAWS labeled_final parquet files found")
    paws_paths = download_repo_files(
        "google-research-datasets/paws", paws_files, RAW_DIR / "paws", endpoint=endpoint
    )

    mrpc_files = list_parquet_under("nyu-mll/glue", endpoint=endpoint, prefix="mrpc")
    if not mrpc_files:
        raise RuntimeError("no GLUE MRPC parquet files found")
    mrpc_paths = download_repo_files(
        "nyu-mll/glue", mrpc_files, RAW_DIR / "glue", endpoint=endpoint
    )

    rows: List[Dict[str, Any]] = []
    for path in paws_paths:
        split = "train"
        name = path.name
        if "validation" in name:
            split = "validation"
        elif "test" in name:
            split = "test"
        for i, ex in enumerate(iter_parquet_rows([path])):
            lab = int(ex["label"])
            s1 = str(ex.get("sentence1") or "").strip()
            s2 = str(ex.get("sentence2") or "").strip()
            if not s1 or not s2:
                continue
            rows.append(
                {
                    "id": f"paws_{split}_{i:06d}",
                    "input": {"text_a": s1, "text_b": s2},
                    "label": {"duplicate": "YES" if lab == 1 else "NO"},
                    "meta": {
                        "source": "google-research-datasets/paws:labeled_final",
                        "split": split,
                        "hard_negative": lab == 0,
                    },
                }
            )

    for path in mrpc_paths:
        split = "train"
        name = path.name
        if "validation" in name:
            split = "validation"
        elif "test" in name:
            split = "test"
        for i, ex in enumerate(iter_parquet_rows([path])):
            lab = int(ex["label"])
            if lab not in (0, 1):
                continue
            s1 = str(ex.get("sentence1") or "").strip()
            s2 = str(ex.get("sentence2") or "").strip()
            if not s1 or not s2:
                continue
            rows.append(
                {
                    "id": f"mrpc_{split}_{i:06d}",
                    "input": {"text_a": s1, "text_b": s2},
                    "label": {"duplicate": "YES" if lab == 1 else "NO"},
                    "meta": {
                        "source": "nyu-mll/glue:mrpc",
                        "split": split,
                        "hard_negative": False,
                    },
                }
            )
    print(f"semantic_dedup full candidates: {len(rows)}")
    return _maybe_limit(rows, limit, seed)


def build_quality_filtering(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    files = list_parquet_under(
        "HuggingFaceFW/fineweb-edu-llama3-annotations", endpoint=endpoint, prefix="data"
    )
    if not files:
        raise RuntimeError("no fineweb-edu annotation shards found")
    paths = download_repo_files(
        "HuggingFaceFW/fineweb-edu-llama3-annotations",
        files,
        RAW_DIR / "fineweb_edu_ann",
        endpoint=endpoint,
    )

    rows: List[Dict[str, Any]] = []
    global_i = 0
    for path in paths:
        for ex in iter_parquet_rows([path]):
            text = str(ex.get("text") or "").strip()
            if not text:
                continue
            score = ex.get("score", ex.get("int_score"))
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            decision = "keep" if score_f >= 3.0 else "remove"
            rows.append(
                {
                    "id": f"fwedu_{global_i:07d}",
                    "input": {"text": _truncate(text, 1200)},
                    "label": {"decision": decision, "score": score_f},
                    "meta": {
                        "source": "HuggingFaceFW/fineweb-edu-llama3-annotations",
                        "shard": path.name,
                        "quality": "high" if decision == "keep" else "low",
                    },
                }
            )
            global_i += 1
    print(f"quality_filtering full candidates: {len(rows)}")
    return _maybe_limit(rows, limit, seed)


_TOPIC_STOP = re.compile(
    r"^(write|create|generate|explain|describe|give|provide|what|how|why|list|summarize|tell)\b[:\s-]*",
    re.I,
)


def _topic_from_instruction(instruction: str) -> str:
    s = (instruction or "").strip()
    s = _TOPIC_STOP.sub("", s)
    s = re.sub(r"\s+", " ", s)
    if len(s) > 80:
        s = s[:77].rstrip() + "..."
    return s or "general knowledge"


def build_instruction_generation(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    remote = "alpaca_data_cleaned.json"
    path = download_file(
        "yahma/alpaca-cleaned",
        remote,
        RAW_DIR / "alpaca" / remote,
        endpoint=endpoint,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(data):
        instruction = str(ex.get("instruction") or "").strip()
        if len(instruction) < 12:
            continue
        topic = _topic_from_instruction(instruction)
        lower = instruction.lower()
        requires_reasoning = any(
            k in lower
            for k in ("why", "reason", "step", "prove", "calculate", "derive", "explain how", "compare")
        )
        if any(k in lower for k in ("simple", "basic", "briefly", "one sentence")):
            difficulty = "easy"
        elif any(k in lower for k in ("advanced", "complex", "rigorous", "prove")):
            difficulty = "hard"
        else:
            difficulty = "medium"
        constraints = {"difficulty": difficulty, "requires_reasoning": requires_reasoning}
        rows.append(
            {
                "id": f"alpaca_{idx:06d}",
                "input": {"topic": topic, "constraints": constraints},
                "label": {
                    "topic": topic,
                    "constraints": constraints,
                    "reference_instruction": instruction,
                    "reference_input": str(ex.get("input") or ""),
                    "reference_output": _truncate(str(ex.get("output") or ""), 800),
                },
                "meta": {"source": "yahma/alpaca-cleaned"},
            }
        )
    print(f"instruction_generation full candidates: {len(rows)}")
    return _maybe_limit(rows, limit, seed)


def _sample_texts(items: List[Dict[str, Any]], keys: Sequence[str], k: int, rng: random.Random) -> List[str]:
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    out: List[str] = []
    for i in idxs:
        ex = items[i]
        text = ""
        for key in keys:
            if ex.get(key):
                text = str(ex[key])
                break
        text = _truncate(text, 160)
        if text:
            out.append(text)
        if len(out) >= k:
            break
    return out


def build_dataset_mixing(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    # Full domain corpora downloads
    alpaca_path = download_file(
        "yahma/alpaca-cleaned",
        "alpaca_data_cleaned.json",
        RAW_DIR / "alpaca" / "alpaca_data_cleaned.json",
        endpoint=endpoint,
    )
    gsm_files = list_parquet_under("openai/gsm8k", endpoint=endpoint, prefix="main")
    gsm_paths = download_repo_files("openai/gsm8k", gsm_files, RAW_DIR / "gsm8k", endpoint=endpoint)
    code_files = list_parquet_under(
        "openai/openai_humaneval", endpoint=endpoint, prefix="openai_humaneval"
    )
    code_paths = download_repo_files(
        "openai/openai_humaneval", code_files, RAW_DIR / "humaneval", endpoint=endpoint
    )
    ann_files = list_parquet_under(
        "HuggingFaceFW/fineweb-edu-llama3-annotations", endpoint=endpoint, prefix="data"
    )
    ann_paths = download_repo_files(
        "HuggingFaceFW/fineweb-edu-llama3-annotations",
        ann_files,
        RAW_DIR / "fineweb_edu_ann",
        endpoint=endpoint,
    )

    alpaca = json.loads(alpaca_path.read_text(encoding="utf-8"))
    gsm8k = list(iter_parquet_rows(gsm_paths))
    code = list(iter_parquet_rows(code_paths))

    # Knowledge pool: all high-score snippets from full annotations (store compact)
    knowledge_hi: List[Dict[str, Any]] = []
    for path in ann_paths:
        for ex in iter_parquet_rows([path]):
            try:
                sc = float(ex.get("score", ex.get("int_score", 0)))
            except Exception:
                continue
            text = str(ex.get("text") or "").strip()
            if sc >= 3.0 and text:
                knowledge_hi.append({"text": text, "score": sc})

    # Persist domain pools for audit / future Level-3
    pools_dir = SEED_DIR / "domain_pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        pools_dir / "instruction_alpaca.jsonl",
        [
            {
                "id": f"inst_{i:06d}",
                "instruction": ex.get("instruction"),
                "input": ex.get("input"),
                "output": _truncate(str(ex.get("output") or ""), 500),
            }
            for i, ex in enumerate(alpaca)
        ],
    )
    _write_jsonl(
        pools_dir / "math_gsm8k.jsonl",
        [
            {
                "id": f"math_{i:06d}",
                "question": ex.get("question"),
                "answer": ex.get("answer"),
            }
            for i, ex in enumerate(gsm8k)
        ],
    )
    _write_jsonl(
        pools_dir / "code_humaneval.jsonl",
        [
            {
                "id": f"code_{i:06d}",
                "task_id": ex.get("task_id"),
                "prompt": ex.get("prompt"),
                "canonical_solution": ex.get("canonical_solution"),
            }
            for i, ex in enumerate(code)
        ],
    )
    _write_jsonl(
        pools_dir / "knowledge_fineweb_edu_hi.jsonl",
        [
            {
                "id": f"know_{i:07d}",
                "text": _truncate(ex["text"], 800),
                "score": ex["score"],
            }
            for i, ex in enumerate(knowledge_hi)
        ],
    )

    rng = random.Random(seed)
    domain_examples = {
        "Instruction": _sample_texts(alpaca, ["instruction"], 20, rng),
        "Math": _sample_texts(gsm8k, ["question"], 20, rng),
        "Code": _sample_texts(code, ["prompt", "canonical_solution"], 20, rng),
        "Knowledge": _sample_texts(knowledge_hi, ["text"], 20, rng),
    }

    scenarios = [
        {
            "target": "maximize multi-step reasoning ability",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 25, "Math": 40, "Code": 20, "Knowledge": 15},
            "budget": 100000,
        },
        {
            "target": "maximize coding ability",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 20, "Math": 15, "Code": 50, "Knowledge": 15},
            "budget": 80000,
        },
        {
            "target": "maximize general knowledge (MMLU-style)",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 25, "Math": 15, "Code": 10, "Knowledge": 50},
            "budget": 120000,
        },
        {
            "target": "balanced general assistant",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 40, "Math": 20, "Code": 20, "Knowledge": 20},
            "budget": 100000,
        },
        {
            "target": "maximize math contest performance",
            "domains": ["Instruction", "Math", "Code"],
            "reference": {"Instruction": 15, "Math": 65, "Code": 20},
            "budget": 60000,
        },
        {
            "target": "improve instruction following under limited data",
            "domains": ["Instruction", "Knowledge"],
            "reference": {"Instruction": 70, "Knowledge": 30},
            "budget": 40000,
        },
        {
            "target": "maximize code + reasoning jointly",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 15, "Math": 30, "Code": 40, "Knowledge": 15},
            "budget": 90000,
        },
        {
            "target": "chat fluency with light reasoning",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 45, "Math": 15, "Code": 15, "Knowledge": 25},
            "budget": 70000,
        },
        {
            "target": "minimize catastrophic forgetting on knowledge while adding code",
            "domains": ["Instruction", "Code", "Knowledge"],
            "reference": {"Instruction": 25, "Code": 30, "Knowledge": 45},
            "budget": 100000,
        },
        {
            "target": "STEM-heavy mixture for reasoning benchmarks",
            "domains": ["Instruction", "Math", "Code", "Knowledge"],
            "reference": {"Instruction": 20, "Math": 35, "Code": 25, "Knowledge": 20},
            "budget": 110000,
        },
    ]

    rows: List[Dict[str, Any]] = []
    idx = 1
    for base in scenarios:
        for factor in (0.5, 1.0, 1.5):
            budget = int(base["budget"] * factor)
            domains = list(base["domains"])
            ref = {d: float(base["reference"][d]) for d in domains}
            rows.append(
                {
                    "id": f"mix_hf_{idx:03d}",
                    "input": {
                        "target": base["target"],
                        "budget": budget,
                        "domains": domains,
                        "domain_examples": {d: domain_examples.get(d, [])[:5] for d in domains},
                        "domain_pool_sizes": {
                            "Instruction": len(alpaca),
                            "Math": len(gsm8k),
                            "Code": len(code),
                            "Knowledge": len(knowledge_hi),
                        },
                    },
                    "label": {"domains": domains, "reference": ref},
                    "meta": {
                        "source_domains": {
                            "Instruction": "yahma/alpaca-cleaned (full)",
                            "Math": "openai/gsm8k main train+test (full)",
                            "Code": "openai/openai_humaneval (full)",
                            "Knowledge": "HuggingFaceFW/fineweb-edu-llama3-annotations score>=3 (full)",
                        }
                    },
                }
            )
            idx += 1
    print(f"dataset_mixing scenarios: {len(rows)}; domain pools written under seed/domain_pools/")
    return _maybe_limit(rows, limit, seed)


def _gsm_answer(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", answer_field.replace(",", ""))
    return m.group(0) if m else answer_field.strip()


def _gsm_difficulty(answer_field: str) -> str:
    steps = answer_field.count("<<")
    if steps <= 1:
        return "easy"
    if steps <= 3:
        return "medium"
    return "hard"


def _load_alpaca(endpoint: str) -> List[Dict[str, Any]]:
    path = download_file(
        "yahma/alpaca-cleaned",
        "alpaca_data_cleaned.json",
        RAW_DIR / "alpaca" / "alpaca_data_cleaned.json",
        endpoint=endpoint,
    )
    return list(json.loads(path.read_text(encoding="utf-8")))


def _load_gsm8k(endpoint: str) -> List[Dict[str, Any]]:
    files = list_parquet_under("openai/gsm8k", endpoint=endpoint, prefix="main")
    paths = download_repo_files("openai/gsm8k", files, RAW_DIR / "gsm8k", endpoint=endpoint)
    return list(iter_parquet_rows(paths))


def _load_fineweb_scored(
    endpoint: str, *, max_rows: Optional[int] = None
) -> List[Dict[str, Any]]:
    files = list_parquet_under(
        "HuggingFaceFW/fineweb-edu-llama3-annotations", endpoint=endpoint, prefix="data"
    )
    paths = download_repo_files(
        "HuggingFaceFW/fineweb-edu-llama3-annotations",
        files,
        RAW_DIR / "fineweb_edu_ann",
        endpoint=endpoint,
    )
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for ex in iter_parquet_rows([path]):
            text = str(ex.get("text") or "").strip()
            if len(text) < 60:
                continue
            try:
                score = float(ex.get("score", ex.get("int_score")))
            except (TypeError, ValueError):
                continue
            rows.append({"text": _truncate(text, 1200), "score": score, "shard": path.name})
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def _load_hendrycks_math(endpoint: str) -> List[Dict[str, Any]]:
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    rows: List[Dict[str, Any]] = []
    for subject in subjects:
        files = list_parquet_under(
            "EleutherAI/hendrycks_math", endpoint=endpoint, prefix=subject
        )
        if not files:
            continue
        paths = download_repo_files(
            "EleutherAI/hendrycks_math",
            files,
            RAW_DIR / "hendrycks_math",
            endpoint=endpoint,
        )
        for path in paths:
            split = "train" if "train" in path.name else "test"
            for i, ex in enumerate(iter_parquet_rows([path])):
                problem = str(ex.get("problem") or "").strip()
                solution = str(ex.get("solution") or "").strip()
                if len(problem) < 20:
                    continue
                level = str(ex.get("level") or "")
                rows.append(
                    {
                        "subject": subject,
                        "problem": problem,
                        "solution": solution,
                        "level": level,
                        "split": split,
                        "idx": i,
                    }
                )
    print(f"hendrycks_math loaded: {len(rows)}")
    return rows


def _corrupt_json_obj(obj: Dict[str, str], rng: random.Random) -> str:
    """Local corruption overlays on HF alpaca rows (needed for repair supervision)."""
    mode = rng.choice(["missing_brace", "missing_field", "trailing_comma", "unquoted", "cut"])
    if mode == "missing_field":
        broken = dict(obj)
        broken.pop("output", None)
        return json.dumps(broken, ensure_ascii=False)[:-1]
    if mode == "trailing_comma":
        return (
            '{\n"instruction": '
            + json.dumps(obj["instruction"], ensure_ascii=False)
            + ',\n"input": '
            + json.dumps(obj.get("input", ""), ensure_ascii=False)
            + ',\n"output": '
            + json.dumps(obj["output"], ensure_ascii=False)
            + ",\n}"
        )
    if mode == "unquoted":
        return "{\ninstruction: " + obj["instruction"][:100] + ",\noutput:\n"
    if mode == "cut":
        good = json.dumps(obj, ensure_ascii=False)
        return good[: max(40, len(good) // 2)]
    return json.dumps(obj, ensure_ascii=False)[:-1]


def build_format_repair(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    alpaca = _load_alpaca(endpoint)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(alpaca):
        obj = {
            "instruction": str(ex.get("instruction") or "").strip(),
            "input": str(ex.get("input") or ""),
            "output": str(ex.get("output") or "").strip(),
        }
        if len(obj["instruction"]) < 8 or len(obj["output"]) < 8:
            continue
        rows.append(
            {
                "id": f"repair_alpaca_{idx:06d}",
                "input": {
                    "broken_json": _corrupt_json_obj(obj, rng),
                    "schema_hint": ["instruction", "input", "output"],
                },
                "label": {
                    "target": obj,
                    "required_keys": ["instruction", "input", "output"],
                },
                "meta": {
                    "source": "yahma/alpaca-cleaned",
                    "corruption": "synthetic_on_hf_row",
                },
            }
        )
    print(f"format_repair candidates: {len(rows)}")
    return _maybe_limit(rows, limit, seed)


def build_quality_ranking(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    scored = _load_fineweb_scored(endpoint, max_rows=120000)
    rng = random.Random(seed)
    bands: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for ex in scored:
        s = int(min(5, max(1, round(ex["score"]))))
        if len(bands[s]) < 800:
            bands[s].append(ex)
    target = limit or 2000
    rows: List[Dict[str, Any]] = []
    for i in range(target):
        chosen = rng.sample([1, 2, 3, 4, 5], k=4)
        cands = []
        for s in chosen:
            pool = bands.get(s) or []
            if not pool:
                continue
            item = rng.choice(pool)
            cid = f"c{i}_s{s}_{abs(hash(item['text'])) % 10_000_000}"
            cands.append({"id": cid, "text": item["text"], "score": float(s)})
        if len(cands) < 3:
            continue
        ranking = [c["id"] for c in sorted(cands, key=lambda x: -x["score"])]
        rows.append(
            {
                "id": f"rank_fwedu_{i:06d}",
                "input": {"candidates": [{"id": c["id"], "text": c["text"]} for c in cands]},
                "label": {
                    "ranking": ranking,
                    "scores": {c["id"]: c["score"] for c in cands},
                },
                "meta": {"source": "HuggingFaceFW/fineweb-edu-llama3-annotations"},
            }
        )
    print(f"quality_ranking candidates: {len(rows)}")
    return rows if limit is None else rows[:limit]


def build_diversity_selection(limit: Optional[int], seed: int, endpoint: str) -> List[Dict[str, Any]]:
    alpaca = _load_alpaca(endpoint)
    rng = random.Random(seed)
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for idx, ex in enumerate(alpaca):
        instr = str(ex.get("instruction") or "").strip()
        if len(instr) < 12:
            continue
        topic = _topic_from_instruction(instr).lower()
        words = re.findall(r"[a-z0-9]+", topic)
        cluster = "_".join(words[:3]) if words else "misc"
        bucket = by_topic.setdefault(cluster, [])
        if len(bucket) < 6:
            bucket.append({"idx": idx, "instruction": instr, "cluster": cluster})
        if len(by_topic) >= 2000 and all(len(v) >= 2 for v in list(by_topic.values())[:50]):
            break
    topics = [t for t, items in by_topic.items() if items]
    target = limit or 2000
    rows: List[Dict[str, Any]] = []
    for i in range(target):
        k_topics = min(6, len(topics))
        if k_topics < 3:
            break
        picked = rng.sample(topics, k=k_topics)
        candidates: List[Dict[str, Any]] = []
        for t in picked:
            for j, item in enumerate(by_topic[t][:2]):
                candidates.append(
                    {
                        "id": f"sel{i}_{item['idx']}_{j}",
                        "instruction": item["instruction"],
                        "text": item["instruction"],
                        "topic": t,
                        "cluster": t,
                    }
                )
        budget = min(k_topics, len(candidates))
        selected: List[str] = []
        seen = set()
        for c in candidates:
            if c["cluster"] in seen:
                continue
            selected.append(c["id"])
            seen.add(c["cluster"])
            if len(selected) >= budget:
                break
        rows.append(
            {
                "id": f"div_alpaca_{i:06d}",
                "input": {"candidates": candidates, "budget": budget},
                "label": {"selected": selected, "budget": budget},
                "meta": {"source": "yahma/alpaca-cleaned"},
            }
        )
    print(f"diversity_selection candidates: {len(rows)}")
    return rows if limit is None else rows[:limit]


def build_reasoning_generation(
    limit: Optional[int], seed: int, endpoint: str
) -> List[Dict[str, Any]]:
    gsm = _load_gsm8k(endpoint)
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(gsm):
        q = str(ex.get("question") or "").strip()
        a_field = str(ex.get("answer") or "")
        if len(q) < 20:
            continue
        rows.append(
            {
                "id": f"reason_gsm8k_{i:06d}",
                "input": {
                    "question": q,
                    "capability": "multi-step arithmetic reasoning",
                },
                "label": {
                    "answer": _gsm_answer(a_field),
                    "reference_solution": a_field,
                },
                "meta": {"source": "openai/gsm8k:main"},
            }
        )
    print(f"reasoning_generation candidates: {len(rows)}")
    return _maybe_limit(rows, limit, seed)


def build_hard_sample_generation(
    limit: Optional[int], seed: int, endpoint: str
) -> List[Dict[str, Any]]:
    """Capability prompts grounded in Hendrycks MATH subjects/problems."""
    math_rows = _load_hendrycks_math(endpoint)
    rng = random.Random(seed)
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for ex in math_rows:
        # Prefer harder levels when annotated like "Level 4"
        level = str(ex.get("level") or "")
        hardish = ("Level 4" in level) or ("Level 5" in level) or (level == "")
        if not hardish and rng.random() > 0.35:
            continue
        by_subject.setdefault(ex["subject"], []).append(ex)
    subjects = [s for s, items in by_subject.items() if len(items) >= 2]
    target = limit or 2000
    rows: List[Dict[str, Any]] = []
    for i in range(target):
        if not subjects:
            break
        subject = subjects[i % len(subjects)]
        pool = by_subject[subject]
        exemplar, reference = rng.sample(pool, k=2)
        capability = subject.replace("_", " ") + " contest problem writing"
        rows.append(
            {
                "id": f"hard_math_{i:06d}",
                "input": {
                    "capability": capability,
                    "hint": (
                        "Write a new hard problem in the same subject. "
                        "Use traps / multi-step structure similar to the exemplar."
                    ),
                    "exemplar_problem": _truncate(exemplar["problem"], 700),
                    "exemplar_level": exemplar.get("level") or "",
                },
                "label": {
                    "reference": {
                        "question": reference["problem"],
                        "answer": _truncate(reference.get("solution") or "", 200),
                        "why_hard": f"MATH {subject} {reference.get('level') or ''}".strip(),
                    }
                },
                "meta": {
                    "source": "EleutherAI/hendrycks_math",
                    "subject": subject,
                    "exemplar_split": exemplar.get("split"),
                },
            }
        )
    print(f"hard_sample_generation candidates: {len(rows)}")
    return rows if limit is None else rows[:limit]


def build_curriculum_scheduling(
    limit: Optional[int], seed: int, endpoint: str
) -> List[Dict[str, Any]]:
    """Item schedules from GSM8K difficulty (step-count heuristic on HF answers)."""
    gsm = _load_gsm8k(endpoint)
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for i, ex in enumerate(gsm):
        q = str(ex.get("question") or "").strip()
        a = str(ex.get("answer") or "")
        if len(q) < 20:
            continue
        diff = _gsm_difficulty(a)
        buckets[diff].append({"id": f"gsm_{i}", "question": q, "difficulty": diff})
    target = limit or 2000
    rows: List[Dict[str, Any]] = []
    for i in range(target):
        items: List[Dict[str, Any]] = []
        for diff in ("easy", "medium", "hard"):
            pool = buckets[diff]
            if len(pool) < 3:
                continue
            for ex in rng.sample(pool, k=3):
                items.append(
                    {
                        "id": f"item_{i}_{ex['id']}",
                        "difficulty": diff,
                        "text": _truncate(ex["question"], 220),
                    }
                )
        if len(items) < 6:
            continue
        order = [
            it["id"]
            for it in sorted(
                items, key=lambda x: {"easy": 0, "medium": 1, "hard": 2}[x["difficulty"]]
            )
        ]
        rows.append(
            {
                "id": f"curr_gsm8k_{i:06d}",
                "input": {
                    "items": items,
                    "training_steps": int(rng.choice([500, 1000, 2000, 5000])),
                    "phases": ["easy", "medium", "hard"],
                    "notes": "Difficulty estimated from GSM8K solution step markers (<< >>).",
                },
                "label": {"mode": "items", "schedule": order},
                "meta": {"source": "openai/gsm8k:main", "difficulty_heuristic": "calc_steps"},
            }
        )
    print(f"curriculum_scheduling candidates: {len(rows)}")
    return rows if limit is None else rows[:limit]


def build_data_utility_prediction(
    limit: Optional[int], seed: int, endpoint: str
) -> List[Dict[str, Any]]:
    scored = _load_fineweb_scored(endpoint, max_rows=None if limit is None else max(limit * 8, 5000))
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(scored):
        rows.append(
            {
                "id": f"util_fwedu_{i:07d}",
                "input": {"text": ex["text"]},
                "label": {"utility": float(ex["score"]), "score": float(ex["score"])},
                "meta": {
                    "source": "HuggingFaceFW/fineweb-edu-llama3-annotations",
                    "shard": ex.get("shard"),
                },
            }
        )
    print(f"data_utility_prediction candidates: {len(rows)}")
    return _maybe_limit(rows, limit, seed)


def build_corpus_filtering(
    limit: Optional[int], seed: int, endpoint: str
) -> List[Dict[str, Any]]:
    """Dirty batch filtering: FineWeb-Edu high/low mix; gold keep = score>=3."""
    scored = _load_fineweb_scored(endpoint, max_rows=80000)
    rng = random.Random(seed)
    hi = [ex for ex in scored if float(ex["score"]) >= 3.0]
    lo = [ex for ex in scored if float(ex["score"]) < 3.0]
    target = limit or 2000
    rows: List[Dict[str, Any]] = []
    for i in range(target):
        n_hi = rng.randint(3, 5)
        n_lo = rng.randint(5, 8)
        if len(hi) < n_hi or len(lo) < n_lo:
            break
        picked_hi = rng.sample(hi, k=n_hi)
        picked_lo = rng.sample(lo, k=n_lo)
        docs = []
        gold_keep = []
        for j, ex in enumerate(picked_hi):
            did = f"d{i}_h{j}"
            docs.append({"id": did, "text": ex["text"]})
            gold_keep.append(did)
        for j, ex in enumerate(picked_lo):
            did = f"d{i}_l{j}"
            docs.append({"id": did, "text": ex["text"]})
        rng.shuffle(docs)
        rows.append(
            {
                "id": f"cfilt_fwedu_{i:06d}",
                "input": {
                    "documents": docs,
                    "budget": len(gold_keep),
                    "notes": "Mixed high/low FineWeb-Edu quality; keep educational pages.",
                },
                "label": {"keep": gold_keep},
                "meta": {
                    "source": "HuggingFaceFW/fineweb-edu-llama3-annotations",
                    "n_docs": len(docs),
                    "n_gold_keep": len(gold_keep),
                },
            }
        )
    print(f"corpus_filtering candidates: {len(rows)}")
    return rows if limit is None else rows[:limit]


def build_corpus_dedup(
    limit: Optional[int], seed: int, endpoint: str
) -> List[Dict[str, Any]]:
    """Corpus batches with PAWS paraphrase pairs planted as near-duplicates."""
    paws_files = list_parquet_under(
        "google-research-datasets/paws", endpoint=endpoint, prefix="labeled_final"
    )
    paws_paths = download_repo_files(
        "google-research-datasets/paws", paws_files, RAW_DIR / "paws", endpoint=endpoint
    )
    rng = random.Random(seed)
    dupe_pairs: List[Tuple[str, str]] = []
    uniques: List[str] = []
    for ex in iter_parquet_rows(paws_paths):
        s1 = str(ex.get("sentence1") or "").strip()
        s2 = str(ex.get("sentence2") or "").strip()
        if len(s1) < 20 or len(s2) < 20:
            continue
        lab = int(ex.get("label", -1))
        if lab == 1:
            dupe_pairs.append((s1, s2))
        elif lab == 0:
            uniques.append(s1)
            if len(uniques) < 20000:
                uniques.append(s2)
        if len(dupe_pairs) >= 8000 and len(uniques) >= 20000:
            break

    target = limit or 2000
    rows: List[Dict[str, Any]] = []
    for i in range(target):
        n_dup_clusters = rng.randint(2, 3)
        n_unique = rng.randint(4, 6)
        if len(dupe_pairs) < n_dup_clusters or len(uniques) < n_unique:
            break
        clusters: List[List[str]] = []
        docs: List[Dict[str, Any]] = []
        gold_remove: List[str] = []
        for c in range(n_dup_clusters):
            a, b = rng.choice(dupe_pairs)
            id_a = f"d{i}_c{c}a"
            id_b = f"d{i}_c{c}b"
            docs.append({"id": id_a, "text": a})
            docs.append({"id": id_b, "text": b})
            clusters.append([id_a, id_b])
            # keep first, remove second
            gold_remove.append(id_b)
        for u in range(n_unique):
            uid = f"d{i}_u{u}"
            docs.append({"id": uid, "text": rng.choice(uniques)})
            clusters.append([uid])
        rng.shuffle(docs)
        rows.append(
            {
                "id": f"cdedup_paws_{i:06d}",
                "input": {
                    "documents": docs,
                    "notes": "Near-duplicates planted from PAWS paraphrase pairs.",
                },
                "label": {
                    "clusters": clusters,
                    "remove": gold_remove,
                    "keep": [d["id"] for d in docs if d["id"] not in set(gold_remove)],
                },
                "meta": {
                    "source": "google-research-datasets/paws:labeled_final",
                    "n_docs": len(docs),
                    "n_dup_clusters": n_dup_clusters,
                },
            }
        )
    print(f"corpus_dedup candidates: {len(rows)}")
    return rows if limit is None else rows[:limit]


ALL_TASK_BUILDERS = {
    "semantic_dedup": build_semantic_dedup,
    "quality_filtering": build_quality_filtering,
    "corpus_filtering": build_corpus_filtering,
    "format_repair": build_format_repair,
    "quality_ranking": build_quality_ranking,
    "diversity_selection": build_diversity_selection,
    "instruction_generation": build_instruction_generation,
    "reasoning_generation": build_reasoning_generation,
    "hard_sample_generation": build_hard_sample_generation,
    "dataset_mixing": build_dataset_mixing,
    "curriculum_scheduling": build_curriculum_scheduling,
    "data_utility_prediction": build_data_utility_prediction,
    "corpus_dedup": build_corpus_dedup,
}


SOURCE_NOTES = {
    "semantic_dedup": [
        "google-research-datasets/paws labeled_final train+validation+test",
        "nyu-mll/glue mrpc train+validation+test",
    ],
    "quality_filtering": ["HuggingFaceFW/fineweb-edu-llama3-annotations all data/ shards"],
    "corpus_filtering": [
        "HuggingFaceFW/fineweb-edu-llama3-annotations mixed dirty batches; keep=score>=3"
    ],
    "format_repair": [
        "yahma/alpaca-cleaned full JSON",
        "local JSON corruption overlays for repair targets",
    ],
    "quality_ranking": [
        "HuggingFaceFW/fineweb-edu-llama3-annotations scores as gold ranking"
    ],
    "diversity_selection": ["yahma/alpaca-cleaned instruction clusters"],
    "instruction_generation": ["yahma/alpaca-cleaned full JSON"],
    "reasoning_generation": ["openai/gsm8k main train+test"],
    "hard_sample_generation": [
        "EleutherAI/hendrycks_math subjects as capabilities + exemplar/reference problems"
    ],
    "dataset_mixing": [
        "domain pools: alpaca + gsm8k main + humaneval + fineweb-edu annotations (score>=3)",
        "scenario prompts in dataset_mixing.jsonl",
    ],
    "curriculum_scheduling": [
        "openai/gsm8k ordered by solution-step difficulty heuristic"
    ],
    "data_utility_prediction": [
        "HuggingFaceFW/fineweb-edu-llama3-annotations score as utility label"
    ],
    "corpus_dedup": [
        "google-research-datasets/paws paraphrase pairs planted into multi-doc corpora"
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Optional cap when materializing seed JSONL. Default: full available labeled set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--hf-endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    endpoint = (args.hf_endpoint or DEFAULT_ENDPOINT).rstrip("/")
    print(f"HF endpoint: {endpoint}")
    print(f"limit_per_task: {args.limit_per_task or 'FULL'}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    _backup_existing_seed()

    wanted = set(ALL_TASK_BUILDERS)
    if args.tasks.strip().lower() != "all":
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        unknown = wanted - set(ALL_TASK_BUILDERS)
        if unknown:
            raise SystemExit(f"unknown tasks: {sorted(unknown)}")

    # Prefer rebuilding existing SOURCES counts for tasks we skip
    prev: Dict[str, Any] = {}
    manifest_path = SEED_DIR / "SOURCES.json"
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    counts: Dict[str, int] = dict(prev.get("counts") or {})
    for name in ALL_TASK_BUILDERS:
        if name not in wanted:
            continue
        print(f"\n=== building {name} ===")
        rows = ALL_TASK_BUILDERS[name](args.limit_per_task, args.seed, endpoint)
        counts[name] = _write_jsonl(SEED_DIR / f"{name}.jsonl", rows)

    sources = dict(prev.get("sources") or {})
    sources.update({k: SOURCE_NOTES[k] for k in wanted if k in SOURCE_NOTES})
    manifest = {
        "mode": "full" if not args.limit_per_task else f"limit={args.limit_per_task}",
        "limit_per_task": args.limit_per_task,
        "seed": args.seed,
        "endpoint": endpoint,
        "counts": counts,
        "tasks": sorted(set(prev.get("tasks") or []) | wanted),
        "raw_dir": str(RAW_DIR),
        "eval_note": "Run full seed with `dpc-bench run` (no --limit). Subsample with `--limit N`.",
        "level1_note": (
            "All tasks use HF-backed rows where possible. format_repair adds local "
            "corruption overlays on alpaca; curriculum difficulty uses GSM8K step markers."
        ),
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote manifest -> {manifest_path}")
    print(json.dumps({k: counts[k] for k in sorted(wanted)}, indent=2))


if __name__ == "__main__":
    main()
