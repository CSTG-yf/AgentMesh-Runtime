from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class StateType(StrEnum):
    TEXT = "text"
    EMBEDDING = "embedding"
    SUMMARY = "summary"
    EVIDENCE = "evidence"
    CODE_RESULT = "code_result"
    BLOB = "blob"
    HIDDEN = "hidden"


class StateRecord(BaseModel):
    state_id: str = Field(default_factory=lambda: f"state-{uuid4().hex[:12]}")
    trace_id: str
    state_type: StateType
    producer: str
    consumers: list[str] = Field(default_factory=list)
    parent_state_refs: list[str] = Field(default_factory=list)
    payload_ref: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    written_to_memory: bool = False

    @property
    def ref(self) -> str:
        return f"state://{self.state_type.value}/{self.state_id}"
