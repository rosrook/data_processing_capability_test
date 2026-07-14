"""Shared dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Sample:
    id: str
    task: str
    input: Dict[str, Any]
    label: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Dict[str, Any], task: str) -> "Sample":
        return cls(
            id=str(row.get("id") or ""),
            task=task,
            input=dict(row.get("input") or {}),
            label=dict(row.get("label") or {}),
            meta=dict(row.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    id: str
    task: str
    model: str
    input: Dict[str, Any]
    label: Dict[str, Any]
    prediction: Optional[Dict[str, Any]]
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRunResult:
    model: str
    task: str
    n: int
    metrics: Dict[str, Any]
    predictions_path: str
    metrics_path: str
    parse_errors: int = 0
    call_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
