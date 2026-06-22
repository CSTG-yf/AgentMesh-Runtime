from typing import Any

import orjson

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
        compact_result = result.model_copy(
            update={
                "result": _compact_protocol_result(
                    result.result,
                    max_chars=self.context.config.protocol.agent_log_output_max_chars,
                )
            }
        )
        self.messages.append(compact_result)
        append_protocol_agent_io(
            paths=self.context.paths,
            trace_id=self.context.trace_id,
            step=len(self.selected_agents) + 1,
            source_agent=source_agent,
            agent=agent.name,
            action=action,
            params=message.params,
            result=compact_result.result,
            state_refs_in=message.state_refs,
            state_refs_out=result.state_refs,
            result_msg_type=compact_result.msg_type.value,
            max_param_chars=self.context.config.protocol.agent_log_param_max_chars,
            max_result_chars=self.context.config.protocol.agent_log_output_max_chars,
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


def _compact_protocol_result(result: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in result.items():
        if key in {"llm_plan", "llm_summary"}:
            compacted[f"{key}_preview"] = _clip_text(str(value), max_chars)
            compacted[f"{key}_chars"] = len(str(value))
            continue
        if key == "codeact_code":
            code = str(value)
            compacted["codeact_code_preview"] = _clip_text(code, max(200, max_chars // 2))
            compacted["codeact_code_chars"] = len(code)
            continue
        if key == "generated_files" and isinstance(value, list):
            compacted[key] = _compact_generated_files(value, max_chars=max_chars)
            continue
        compacted[str(key)] = _compact_value(value, max_chars=max_chars)
    return compacted


def _compact_generated_files(files: list[Any], *, max_chars: int) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", ""))
        compacted.append(
            {
                "path": item.get("path", ""),
                "language": item.get("language", ""),
                "content_preview": _clip_text(content, max(120, max_chars // 4)),
                "content_chars": len(content),
            }
        )
    return compacted


def _compact_value(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str):
        return _clip_text(value, max_chars)
    if isinstance(value, list):
        return [_compact_value(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        compacted = {
            str(key): _compact_value(item, max_chars=max_chars)
            for key, item in value.items()
        }
        if _json_chars(compacted) <= max_chars * 3:
            return compacted
        return {
            str(key): _compact_value(item, max_chars=max(80, max_chars // 2))
            for key, item in value.items()
            if key not in {"code", "content", "stdout", "stderr"}
        }
    return value


def _clip_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)]}..."


def _json_chars(value: Any) -> int:
    try:
        return len(orjson.dumps(value).decode("utf-8"))
    except TypeError:
        return len(str(value))
