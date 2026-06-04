from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import orjson

from agentmesh.core import rust_available, rust_core
from agentmesh.protocol.codec import encode_payload_compact
from agentmesh.protocol.schema import AMPMessage


@dataclass(frozen=True)
class TypedEnvelopeStats:
    session_dictionary_bytes: int
    typed_envelope_bytes: int
    typed_payload_bytes: int

    @property
    def wire_bytes(self) -> int:
        return self.typed_envelope_bytes


def measure_typed_envelopes(messages: list[AMPMessage]) -> TypedEnvelopeStats:
    dictionary = build_session_dictionary(messages)
    dictionary_bytes = len(encode_payload_compact(dictionary))
    envelope_bytes = 0
    payload_bytes = 0
    for index, message in enumerate(messages, start=1):
        envelope, payload = build_typed_envelope(message, dictionary, payload_index=index)
        envelope_bytes += len(encode_typed_envelope(envelope))
        if payload is not None:
            payload_bytes += len(encode_payload_compact(payload))
    return TypedEnvelopeStats(
        session_dictionary_bytes=dictionary_bytes,
        typed_envelope_bytes=envelope_bytes,
        typed_payload_bytes=payload_bytes,
    )


def encode_typed_envelope(envelope: dict[str, Any]) -> bytes:
    if rust_available() and hasattr(rust_core(), "encode_typed_envelope_bytes"):
        return bytes(rust_core().encode_typed_envelope_bytes(orjson.dumps(envelope).decode()))
    return encode_payload_compact(envelope)


def decode_typed_envelope(data: bytes) -> dict[str, Any]:
    if rust_available() and hasattr(rust_core(), "decode_typed_envelope_json_text"):
        decoded = rust_core().decode_typed_envelope_json_text(data)
    else:
        decoded = rust_core().decode_msgpack_json_text(data) if rust_available() else data
    loaded = orjson.loads(decoded)
    if not isinstance(loaded, dict):
        raise ValueError("typed envelope must decode to an object")
    return loaded


def build_session_dictionary(messages: list[AMPMessage]) -> dict[str, dict[str, int]]:
    agents: set[str] = set()
    actions: set[str] = set()
    capabilities: set[str] = set()
    msg_types: set[str] = set()
    state_refs: set[str] = set()
    traces: set[str] = set()
    for message in messages:
        traces.add(message.trace_id)
        agents.add(message.source_agent)
        agents.add(message.target_agent)
        msg_types.add(message.msg_type.value)
        if message.action:
            actions.add(message.action)
        capabilities.update(message.capability)
        state_refs.update(message.state_refs)
    return {
        "traces": _numbered(traces),
        "agents": _numbered(agents),
        "msg_types": _numbered(msg_types),
        "actions": _numbered(actions),
        "capabilities": _numbered(capabilities),
        "state_refs": _numbered(state_refs),
    }


def build_typed_envelope(
    message: AMPMessage,
    dictionary: dict[str, dict[str, int]],
    *,
    payload_index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = _payload_for(message)
    envelope: dict[str, Any] = {
        "i": payload_index,
        "q": dictionary["traces"][message.trace_id],
        "s": dictionary["agents"][message.source_agent],
        "t": dictionary["agents"][message.target_agent],
        "m": dictionary["msg_types"][message.msg_type.value],
    }
    if message.action:
        envelope["a"] = dictionary["actions"][message.action]
    if message.capability:
        envelope["c"] = [dictionary["capabilities"][item] for item in message.capability]
    if message.state_refs:
        envelope["r"] = [dictionary["state_refs"][item] for item in message.state_refs]
    if payload is not None:
        envelope["p"] = _payload_id(payload_index, payload)
    return envelope, payload


def _payload_for(message: AMPMessage) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if message.params:
        payload["params"] = message.params
    if message.result:
        payload["result"] = message.result
    return payload or None


def _payload_id(index: int, payload: dict[str, Any]) -> str:
    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return f"p{index:x}-{len(encoded):x}"


def _numbered(values: set[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(sorted(values), start=1)}
