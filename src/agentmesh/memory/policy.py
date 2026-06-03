from pydantic import BaseModel

from agentmesh.memory.schema import MemoryUnit


class MemoryWritePolicy(BaseModel):
    min_confidence: float = 0.3

    def should_write(self, unit: MemoryUnit) -> bool:
        return bool(unit.summary.strip()) and unit.confidence >= self.min_confidence
