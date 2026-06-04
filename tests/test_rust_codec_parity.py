import orjson

from agentmesh.core import rust_core
from agentmesh.protocol.codec import decode_message, encode_message
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.state.refs import parse_state_ref


def test_rust_state_ref_parser_returns_parts() -> None:
    state_type, state_id = rust_core().parse_state_ref_parts("state://text/state-abc")

    assert state_type == "text"
    assert state_id == "state-abc"


def test_rust_json_codec_helpers_roundtrip_json() -> None:
    payload = {"b": 2, "a": ["x"]}
    encoded = rust_core().encode_json_bytes(orjson.dumps(payload).decode("utf-8"))
    decoded = orjson.loads(rust_core().decode_json_text(encoded))

    assert decoded == payload


def test_codec_roundtrip_still_returns_amp_message() -> None:
    message = AMPMessage(
        trace_id="trace-1",
        source_agent="a",
        target_agent="b",
        msg_type=MsgType.INVOKE,
        action="plan.create",
        state_refs=["state://text/state-abc"],
    )

    decoded = decode_message(encode_message(message))

    assert decoded.trace_id == "trace-1"
    assert decoded.action == "plan.create"


def test_parse_state_ref_accepts_valid_ref() -> None:
    parsed = parse_state_ref("state://text/state-abc")

    assert parsed.state_type == "text"
    assert parsed.state_id == "state-abc"
