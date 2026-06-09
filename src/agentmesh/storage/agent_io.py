from __future__ import annotations

from typing import Any

from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


def append_text_agent_io(
    *,
    paths: RuntimePaths,
    trace_id: str,
    step: int,
    agent: str,
    target_agent: str,
    input_content: str,
    output_content: str,
) -> None:
    append_jsonl(
        paths.text_agent_io,
        {
            "mode": "text",
            "trace_id": trace_id,
            "step": step,
            "agent": agent,
            "source_agent": agent,
            "target_agent": target_agent,
            "input": {
                "content": input_content,
                "content_chars": len(input_content),
                "content_bytes": len(input_content.encode("utf-8")),
            },
            "output": {
                "content": output_content,
                "content_chars": len(output_content),
                "content_bytes": len(output_content.encode("utf-8")),
            },
            "state_refs_in": [],
            "state_refs_out": [],
            "transport": {
                "communication_model": "plain_text_full_context",
                "message_type": "natural_language_handoff",
            },
        },
    )


def append_protocol_agent_io(
    *,
    paths: RuntimePaths,
    trace_id: str,
    step: int,
    source_agent: str,
    agent: str,
    action: str | None,
    params: dict[str, Any],
    result: dict[str, Any],
    state_refs_in: list[str],
    state_refs_out: list[str],
    result_msg_type: str,
) -> None:
    append_jsonl(
        paths.protocol_agent_io,
        {
            "mode": "protocol",
            "trace_id": trace_id,
            "step": step,
            "agent": agent,
            "source_agent": source_agent,
            "target_agent": agent,
            "input": {
                "action": action,
                "params": params,
                "state_refs": state_refs_in,
            },
            "output": {
                "msg_type": result_msg_type,
                "result": result,
                "state_refs": state_refs_out,
            },
            "state_refs_in": state_refs_in,
            "state_refs_out": state_refs_out,
            "transport": {
                "communication_model": "amp_state_ref",
                "message_type": "INVOKE_RESULT",
            },
        },
    )
