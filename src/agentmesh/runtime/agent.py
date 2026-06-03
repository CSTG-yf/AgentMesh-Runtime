from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage

if TYPE_CHECKING:
    from agentmesh.runtime.registry import RuntimeContext


class BaseAgent(ABC):
    name: str
    capabilities: list[str]

    def hello(self, trace_id: str) -> AMPMessage:
        return AMPMessage(
            trace_id=trace_id,
            source_agent=self.name,
            target_agent="runtime",
            msg_type=MsgType.HELLO,
            result={"agent": self.name},
            capability=self.capabilities,
        )

    def advertise_capabilities(self, trace_id: str) -> AMPMessage:
        return AMPMessage(
            trace_id=trace_id,
            source_agent=self.name,
            target_agent="runtime",
            msg_type=MsgType.CAPABILITY_ADVERTISE,
            result={"agent": self.name, "capabilities": self.capabilities},
            capability=self.capabilities,
        )

    @abstractmethod
    def handle(self, message: AMPMessage, context: RuntimeContext) -> AMPMessage:
        raise NotImplementedError
