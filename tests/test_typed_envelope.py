from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.envelope import (
    build_session_dictionary,
    build_typed_envelope,
    decode_typed_envelope,
    encode_typed_envelope,
    measure_typed_envelopes,
)
from agentmesh.protocol.schema import AMPMessage


def test_typed_envelope_uses_dictionary_ids_and_payload_ref() -> None:
    message = AMPMessage(
        trace_id="trace-1",
        source_agent="runtime",
        target_agent="planner",
        msg_type=MsgType.INVOKE,
        action="plan.create",
        params={"task": "large text lives in payload store"},
        state_refs=["state://text/state-abc", "state://embedding/state-def"],
    )
    dictionary = build_session_dictionary([message])

    envelope, payload = build_typed_envelope(message, dictionary, payload_index=1)

    assert envelope["s"] == dictionary["agents"]["runtime"]
    assert envelope["t"] == dictionary["agents"]["planner"]
    assert envelope["a"] == dictionary["actions"]["plan.create"]
    assert envelope["r"] == [
        dictionary["state_refs"]["state://text/state-abc"],
        dictionary["state_refs"]["state://embedding/state-def"],
    ]
    assert envelope["p"].startswith("p1-")
    assert payload == {"params": {"task": "large text lives in payload store"}}


def test_typed_envelope_wire_is_smaller_than_payload_bytes() -> None:
    messages = [
        AMPMessage(
            trace_id="trace-1",
            source_agent="runtime",
            target_agent="planner",
            msg_type=MsgType.INVOKE,
            action="plan.create",
            params={"task": "x" * 512},
            state_refs=["state://text/state-abc"],
        )
    ]

    stats = measure_typed_envelopes(messages)

    assert stats.session_dictionary_bytes > 0
    assert stats.typed_envelope_bytes > 0
    assert stats.typed_payload_bytes > stats.typed_envelope_bytes


def test_typed_envelope_encode_decode_roundtrip() -> None:
    envelope = {"i": 1, "q": 1, "s": 2, "t": 3, "m": 4, "a": 5, "r": [6, 7]}

    encoded = encode_typed_envelope(envelope)
    decoded = decode_typed_envelope(encoded)

    assert decoded == envelope
