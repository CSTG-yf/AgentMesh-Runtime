from pydantic import BaseModel

from agentmesh.core import rust_available, rust_core
from agentmesh.errors import ProtocolError
from agentmesh.state.schema import StateType


class ParsedStateRef(BaseModel):
    state_type: StateType
    state_id: str


def make_state_ref(state_type: StateType, state_id: str) -> str:
    return f"state://{state_type.value}/{state_id}"


def parse_state_ref(ref: str) -> ParsedStateRef:
    if rust_available():
        try:
            state_type, state_id = rust_core().parse_state_ref_parts(ref)
        except Exception as exc:
            raise ProtocolError(str(exc)) from exc
        return ParsedStateRef(state_type=StateType(state_type), state_id=state_id)
    prefix = "state://"
    if not ref.startswith(prefix):
        raise ProtocolError("StateRef must start with state://")
    parts = ref[len(prefix) :].split("/", maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise ProtocolError("StateRef must use state://<state_type>/<state_id>")
    return ParsedStateRef(state_type=StateType(parts[0]), state_id=parts[1])
