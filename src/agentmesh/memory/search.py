from pydantic import BaseModel

from agentmesh.memory.schema import MemoryUnit


class MemorySearchResult(BaseModel):
    memory: MemoryUnit
    score: float
    semantic_similarity: float = 0.0
    validity_score: float = 0.0
    confidence_score: float = 0.0
    reuse_score: float = 0.0
    recency_score: float = 0.0
    tag_overlap_score: float = 0.0
    reason: str = ""
