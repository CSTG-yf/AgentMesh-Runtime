from agentmesh.errors import ProtocolError
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.protocol.transport import AgentTransport, InProcTransport, TransportMetrics
from agentmesh.runtime.registry import AgentRegistry, RuntimeContext
from agentmesh.storage.agent_io import append_protocol_agent_io


class SyncScheduler:
    def run(self, messages: list[AMPMessage]) -> list[AMPMessage]:
        return messages


class ProtocolScheduler:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        context: RuntimeContext,
        messages: list[AMPMessage],
        protocol_map: dict[str, str] | None = None,
        transport: AgentTransport | None = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.messages = messages
        self.protocol_map = dict(protocol_map or {})
        self.transport = transport or InProcTransport(registry=registry, context=context)
        self.selected_agents: list[str] = []

    def invoke(
        self,
        *,
        source_agent: str,
        action: str,
        params: dict[str, object] | None = None,
        state_refs: list[str] | None = None,
        target_agent: str | None = None,
    ) -> AMPMessage:
        self._validate_action(action)
        agent = (
            self.registry.get(target_agent)
            if target_agent is not None
            else self.registry.require_capability(action)
        )
        message = AMPMessage(
            trace_id=self.context.trace_id,
            source_agent=source_agent,
            target_agent=agent.name,
            msg_type=MsgType.INVOKE,
            action=action,
            params=dict(params or {}),
            state_refs=state_refs or [],
        )
        self.messages.append(message)
        result = self.transport.send(message)
        self.messages.append(result)
        append_protocol_agent_io(
            paths=self.context.paths,
            trace_id=self.context.trace_id,
            step=len(self.selected_agents) + 1,
            source_agent=source_agent,
            agent=agent.name,
            action=action,
            params=message.params,
            result=result.result,
            state_refs_in=message.state_refs,
            state_refs_out=result.state_refs,
            result_msg_type=result.msg_type.value,
        )
        self.selected_agents.append(agent.name)
        return result

    def transport_metrics(self) -> TransportMetrics:
        return self.transport.metrics()

    def _validate_action(self, action: str) -> None:
        if not self.protocol_map:
            return
        if action in set(self.protocol_map.values()):
            return
        raise ProtocolError(f"Action {action} is not allowed by the protocol map")
