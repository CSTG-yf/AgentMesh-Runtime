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
    embedding_vector: list[float] | None = Field(default=None, exclude=True)
    reuse_count: int = 0
    confidence: float = 0.0
    validity_score: float = 0.0
    reuse_policy: str = "verify"
    provenance_trace_id: str
    status: str = "active"
    importance_score: float = 0.0
    last_compacted_at: datetime | None = None
    archive_reason: str | None = None
    source_memory_ids: list[str] = Field(default_factory=list)
    memory_type: str = "task_summary"
    domain: str = "general"
    write_scope: str = "run"
