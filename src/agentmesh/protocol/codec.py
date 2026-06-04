from typing import Any

import orjson

from agentmesh.core import rust_available, rust_core
from agentmesh.protocol.schema import AMPMessage


def encode_message(message: AMPMessage) -> bytes:
    if rust_available():
        return bytes(rust_core().encode_json_bytes(message.model_dump_json()))
    return orjson.dumps(message.model_dump(mode="json"))


def decode_message(data: bytes | str) -> AMPMessage:
    if rust_available():
        payload = data if isinstance(data, bytes) else data.encode("utf-8")
        rust_raw: Any = orjson.loads(rust_core().decode_json_text(payload))
        return AMPMessage.model_validate(rust_raw)
    python_raw: Any = orjson.loads(data)
    return AMPMessage.model_validate(python_raw)
