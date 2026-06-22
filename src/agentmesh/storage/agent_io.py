from __future__ import annotations

from typing import Any

import orjson

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
    max_param_chars: int = 500,
    max_result_chars: int = 1200,
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
                "params": _compact_payload(params, max_chars=max_param_chars),
                "state_refs": state_refs_in,
            },
            "output": {
                "msg_type": result_msg_type,
                "result": _compact_payload(result, max_chars=max_result_chars),
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


def _compact_payload(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str):
        return _clip_text(value, max_chars)
    if isinstance(value, list):
        return [_compact_payload(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        compacted = {
            str(key): _compact_payload(item, max_chars=max_chars)
            for key, item in value.items()
        }
        if _json_chars(compacted) <= max_chars * 3:
            return compacted
        return {
            str(key): _compact_payload(item, max_chars=max(80, max_chars // 2))
            for key, item in value.items()
            if key not in {"llm_plan", "llm_summary", "codeact_code", "generated_files"}
        }
    return value


def _clip_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)]}..."


def _json_chars(value: Any) -> int:
    try:
        return len(orjson.dumps(value).decode("utf-8"))
    except TypeError:
        return len(str(value))
