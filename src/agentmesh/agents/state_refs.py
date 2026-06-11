from __future__ import annotations

from typing import Any, cast

import orjson

from agentmesh.runtime.registry import RuntimeContext
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore


def _resolve_store(context: RuntimeContext) -> StateStore:
    """Use injected context.state_store when available, otherwise create one."""
    if context.state_store is not None:
        return cast(StateStore, context.state_store)
    return StateStore(
        context.paths,
        payload_backend=context.config.state.payload_backend,
        shm_threshold_bytes=context.config.state.shm_threshold_bytes,
    )


def read_state_payloads(
    *,
    context: RuntimeContext,
    state_refs: list[str],
    consumer: str,
) -> list[tuple[str, StateType, Any]]:
    store = _resolve_store(context)
    owned = store is not context.state_store
    payloads: list[tuple[str, StateType, Any]] = []
    try:
        for ref in state_refs:
            record, payload = store.get(ref)
            store.add_consumer(ref, consumer)
            payloads.append((ref, record.state_type, payload))
    finally:
        if owned:
            store.close()
    return payloads


def first_text_payload(
    *,
    context: RuntimeContext,
    state_refs: list[str],
    consumer: str,
) -> str:
    for _ref, state_type, payload in read_state_payloads(
        context=context,
        state_refs=state_refs,
        consumer=consumer,
    ):
        if state_type in {StateType.TEXT, StateType.SUMMARY} and isinstance(payload, str):
            return payload
    return ""


def state_payloads_as_text(
    *,
    context: RuntimeContext,
    state_refs: list[str],
    consumer: str,
) -> str:
    sections: list[str] = []
    for ref, state_type, payload in read_state_payloads(
        context=context,
        state_refs=state_refs,
        consumer=consumer,
    ):
        if isinstance(payload, str):
            rendered = payload
        else:
            rendered = orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")
        sections.append(f"{ref} ({state_type.value}):\n{rendered}")
    return "\n\n".join(sections)


def first_code_result_text(
    *,
    context: RuntimeContext,
    state_refs: list[str],
    consumer: str,
) -> str:
    for _ref, state_type, payload in read_state_payloads(
        context=context,
        state_refs=state_refs,
        consumer=consumer,
    ):
        if state_type != StateType.CODE_RESULT or not isinstance(payload, dict):
            continue
        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        exit_code = payload.get("exit_code", "")
        return (
            f"sandbox exit {exit_code}; "
            f"stdout={stdout or '<empty>'}; "
            f"stderr={stderr or '<empty>'}"
        )
    return ""
