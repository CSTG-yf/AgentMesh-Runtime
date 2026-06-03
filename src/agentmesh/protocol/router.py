from agentmesh.protocol.schema import AMPMessage
from agentmesh.runtime.registry import AgentRegistry, RuntimeContext


class ProtocolRouter:
    def __init__(self, registry: AgentRegistry, context: RuntimeContext) -> None:
        self._registry = registry
        self._context = context

    def route(self, message: AMPMessage) -> AMPMessage:
        return self._registry.get(message.target_agent).handle(message, self._context)
