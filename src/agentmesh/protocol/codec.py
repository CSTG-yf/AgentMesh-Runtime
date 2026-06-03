from typing import Any

import orjson

from agentmesh.protocol.schema import AMPMessage


def encode_message(message: AMPMessage) -> bytes:
    return orjson.dumps(message.model_dump(mode="json"))


def decode_message(data: bytes | str) -> AMPMessage:
    raw: Any = orjson.loads(data)
    return AMPMessage.model_validate(raw)
