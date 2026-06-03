from pydantic import BaseModel

from agentmesh.memory.schema import MemoryUnit


class MemorySearchResult(BaseModel):
    memory: MemoryUnit
    score: float
