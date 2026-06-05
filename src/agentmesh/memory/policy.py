from pydantic import BaseModel

from agentmesh.memory.schema import MemoryUnit


class MemoryWritePolicy(BaseModel):
    min_confidence: float = 0.3
    long_term_min_importance: float = 0.62
    long_term_min_validity: float = 0.55
    min_summary_chars: int = 24

    def should_write(self, unit: MemoryUnit) -> bool:
        return (
            len(unit.summary.strip()) >= self.min_summary_chars
            and unit.confidence >= self.min_confidence
            and unit.validity_score > 0
        )

    def should_write_long_term(self, unit: MemoryUnit) -> bool:
        if not self.should_write(unit):
            return False
        if unit.status != "active":
            return False
        if unit.validity_score < self.long_term_min_validity:
            return False
        if unit.importance_score < self.long_term_min_importance:
            return False
        if unit.memory_type in {"debug_noise", "transient"}:
            return False
        return True
