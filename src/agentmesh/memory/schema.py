from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryUnit(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem-{uuid4().hex[:12]}")
    source_agent: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    task_topic: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    state_refs: list[str] = Field(default_factory=list)
    embedding_ref: str | None = None
    reuse_count: int = 0
    confidence: float = 0.0
    validity_score: float = 0.0
    reuse_policy: str = "verify"
    provenance_trace_id: str
