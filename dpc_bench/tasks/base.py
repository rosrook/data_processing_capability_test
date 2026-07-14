"""Task protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional  # Dict used by stratified loader

from ..io import read_jsonl
from ..models import Sample
from ..paths import DEFAULT_SEED_DIR


class Task(ABC):
    name: str
    seed_filename: str
    required_keys: List[str]
    system_prompt: str

    def seed_path(self, seed_dir: Optional[Path] = None) -> Path:
        base = seed_dir or DEFAULT_SEED_DIR
        return base / self.seed_filename

    def _label_bucket(self, sample: Sample) -> str:
        lab = sample.label or {}
        if "duplicate" in lab:
            return str(lab.get("duplicate"))
        if "decision" in lab:
            return str(lab.get("decision"))
        return "all"

    def load_samples(self, seed_dir: Optional[Path] = None, limit: Optional[int] = None) -> List[Sample]:
        path = self.seed_path(seed_dir)
        if limit is None:
            rows = read_jsonl(path)
            return [Sample.from_dict(row, task=self.name) for row in rows if row.get("id")]

        cap = max(0, int(limit))
        if cap == 0:
            return []

        # Stratified / round-robin over label buckets so --limit N is representative
        # on large full seeds (avoid first-N being all one class).
        from ..io import iter_jsonl

        buckets: Dict[str, List[Sample]] = {}
        for row in iter_jsonl(path):
            if not row.get("id"):
                continue
            sample = Sample.from_dict(row, task=self.name)
            buckets.setdefault(self._label_bucket(sample), []).append(sample)
            n_keys = max(1, len(buckets))
            need_each = cap // n_keys + 1
            if len(buckets) > 1 and all(len(v) >= need_each for v in buckets.values()):
                break

        keys = sorted(buckets)
        if not keys:
            return []
        per = max(1, cap // len(keys))
        selected: List[Sample] = []
        for key in keys:
            selected.extend(buckets[key][:per])
        if len(selected) < cap:
            for key in keys:
                already = sum(1 for s in selected if self._label_bucket(s) == key)
                need = cap - len(selected)
                if need <= 0:
                    break
                selected.extend(buckets[key][already: already + need])
        return selected[:cap]

    @abstractmethod
    def build_user_prompt(self, sample: Sample) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_prediction(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def dry_run_prediction(self, sample: Sample) -> Dict[str, Any]:
        """Deterministic correct-ish answer for offline smoke."""
        return dict(sample.label)
