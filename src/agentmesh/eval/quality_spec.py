from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QualitySpec(BaseModel):
    version: str = "1.0"
    rule_id: str
    kind: Literal["non_empty", "contains_all", "regex"]
    expected: list[str] = Field(default_factory=list)
    pattern: str = ""
    pass_threshold: float = 1.0

    @model_validator(mode="after")
    def validate_rule(self) -> QualitySpec:
        if self.kind == "contains_all" and not self.expected:
            raise ValueError("contains_all requires expected values")
        if self.kind == "regex" and not self.pattern:
            raise ValueError("regex requires pattern")
        return self


class QualityResult(BaseModel):
    rule_id: str = "unscored"
    scored: bool = False
    passed: bool = False
    score: float | None = None
    matched: int = 0
    total: int = 0
    reason: str = ""


def evaluate_quality(answer: str, spec: QualitySpec | None) -> QualityResult:
    if spec is None:
        return QualityResult(reason="quality rule not configured")
    if spec.kind == "non_empty":
        score = 1.0 if answer.strip() else 0.0
        return _result(spec, score, int(score), 1)
    if spec.kind == "regex":
        matched = int(re.search(spec.pattern, answer, flags=re.IGNORECASE) is not None)
        return _result(spec, float(matched), matched, 1)
    lowered = answer.casefold()
    matched = sum(item.casefold() in lowered for item in spec.expected)
    total = len(spec.expected)
    return _result(spec, matched / total, matched, total)


def _result(
    spec: QualitySpec,
    score: float,
    matched: int,
    total: int,
) -> QualityResult:
    return QualityResult(
        rule_id=spec.rule_id,
        scored=True,
        passed=score >= spec.pass_threshold,
        score=score,
        matched=matched,
        total=total,
        reason=f"matched {matched}/{total}",
    )
