from datetime import UTC

import pytest

from agentmesh.errors import ProtocolError
from agentmesh.protocol.codec import decode_message, encode_message
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage


def test_amp_message_defaults_and_roundtrip() -> None:
    message = AMPMessage(
        trace_id="trace-1",
        source_agent="planner",
        target_agent="retriever",
        msg_type=MsgType.INVOKE,
        action="memory.keyword_search",
        params={"query": "state passing"},
        state_refs=["state://text/state-abc123"],
    )

    assert message.message_id.startswith("msg-")
    assert len(message.message_id) == 16
    assert message.created_at.tzinfo == UTC

    encoded = encode_message(message)
    decoded = decode_message(encoded)

    assert decoded == message


@pytest.mark.parametrize(
    ("msg_type", "kwargs", "error"),
    [
        (MsgType.INVOKE, {"action": None}, "action"),
        (MsgType.RESULT, {"result": {}}, "result"),
        (MsgType.ERROR, {"result": {"error_code": "X"}}, "error_code and message"),
    ],
)
def test_amp_message_validates_required_fields(
    msg_type: MsgType, kwargs: dict[str, object], error: str
) -> None:
    with pytest.raises(ProtocolError, match=error):
        AMPMessage(
            trace_id="trace-1",
            source_agent="planner",
            target_agent="retriever",
            msg_type=msg_type,
            **kwargs,
        )


def test_amp_message_rejects_non_state_refs() -> None:
    with pytest.raises(ProtocolError, match="state://"):
        AMPMessage(
            trace_id="trace-1",
            source_agent="planner",
            target_agent="retriever",
            msg_type=MsgType.INVOKE,
            action="state.get",
            state_refs=["text/state-abc123"],
        )
