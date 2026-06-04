from pathlib import Path

from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_state_store_can_reload_record_from_sqlite_index(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    first = StateStore(paths)
    ref = first.put_text("trace-1", "tester", "hello", metadata={"k": "v"})
    paths.protocol_states.unlink()

    second = StateStore(paths)
    record, payload = second.get(ref)

    assert payload == "hello"
    assert record.trace_id == "trace-1"
    assert record.metadata == {"k": "v"}
