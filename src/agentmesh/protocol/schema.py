from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentmesh.errors import ProtocolError
from agentmesh.protocol.enums import MsgType


class AMPMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg-{uuid4().hex[:12]}")
    trace_id: str
    source_agent: str
    target_agent: str
    msg_type: MsgType
    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    capability: list[str] = Field(default_factory=list)
    state_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: Any) -> None:
        if self.msg_type == MsgType.INVOKE and not self.action:
            raise ProtocolError("INVOKE messages require action")
        if self.msg_type == MsgType.RESULT and not self.result:
            raise ProtocolError("RESULT messages require result")
        if self.msg_type == MsgType.ERROR:
            if not {"error_code", "message"}.issubset(self.result):
                raise ProtocolError("ERROR result requires error_code and message")
        invalid_refs = [ref for ref in self.state_refs if not ref.startswith("state://")]
        if invalid_refs:
            raise ProtocolError("state_refs must start with state://")
