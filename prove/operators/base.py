"""Shared operator result schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OperatorResult:
    name: str
    score: float  # hallucination evidence in [0, 1]
    predicted_label: int  # 1 if score > 0.5
    parse_success: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def failed_result(name: str, error: str, step: str = "") -> OperatorResult:
    return OperatorResult(
        name=name,
        score=0.0,
        predicted_label=0,
        parse_success=False,
        details={"error": error, "step": step},
    )
