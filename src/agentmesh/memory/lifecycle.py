from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from agentmesh.memory.schema import MemoryUnit


class MemoryLifecycleDecision(BaseModel):
    action: str
    reason: str
    importance_score: float


class MemoryLifecyclePolicy(BaseModel):
    archive_after_days: int = 14
    low_value_threshold: float = 0.35
    keep_threshold: float = 0.75

    def decide(self, unit: MemoryUnit) -> MemoryLifecycleDecision:
        importance = unit.importance_score or self._importance(unit)
        if unit.status != "active":
            return MemoryLifecycleDecision(
                action="skip",
                reason=f"status={unit.status}",
                importance_score=importance,
            )
        if importance >= self.keep_threshold:
            return MemoryLifecycleDecision(
                action="keep",
                reason="high importance",
                importance_score=importance,
            )
        if self._age_days(unit) >= self.archive_after_days and unit.reuse_count == 0:
            return MemoryLifecycleDecision(
                action="archive",
                reason="old and never reused",
                importance_score=importance,
            )
        if importance < self.low_value_threshold and unit.reuse_count == 0:
            return MemoryLifecycleDecision(
                action="archive",
                reason="low value and never reused",
                importance_score=importance,
            )
        return MemoryLifecycleDecision(
            action="keep",
            reason="within active retention window",
            importance_score=importance,
        )

    def _importance(self, unit: MemoryUnit) -> float:
        evidence_score = min(1.0, (len(unit.evidence_refs) + len(unit.state_refs)) / 4)
        reuse_score = min(1.0, unit.reuse_count / 5)
        return (
            0.35 * unit.validity_score
            + 0.25 * unit.confidence
            + 0.20 * reuse_score
            + 0.10 * self._recency(unit)
            + 0.10 * evidence_score
        )

    def _age_days(self, unit: MemoryUnit) -> float:
        return max(0.0, (datetime.now(UTC) - unit.created_at).total_seconds() / 86400)

    def _recency(self, unit: MemoryUnit) -> float:
        age = self._age_days(unit)
        return max(0.0, 1.0 - min(1.0, age / self.archive_after_days))
