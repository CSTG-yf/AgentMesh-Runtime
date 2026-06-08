from pathlib import Path

import pytest

from agentmesh.errors import StateNotFoundError
from agentmesh.state.refs import parse_state_ref
from agentmesh.state.schema import StateType
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_state_store_writes_reads_and_tracks_lineage(tmp_path: Path) -> None:
    store = StateStore(RuntimePaths(root=tmp_path))

    text_ref = store.put_text(
        trace_id="trace-1",
        producer="planner",
        text="structured state passing",
        metadata={"topic": "protocol"},
    )
    evidence_ref = store.put_evidence(
        trace_id="trace-1",
        producer="retriever",
        evidence=[{"title": "doc", "snippet": "StateRef avoids repeated context."}],
        parent_state_refs=[text_ref],
    )

    parsed = parse_state_ref(evidence_ref)
    record, payload = store.get(evidence_ref)

    assert store.paths.state_index.exists()
    assert parsed.state_type == StateType.EVIDENCE
    assert record.producer == "retriever"
    assert record.parent_state_refs == [text_ref]
    assert payload[0]["title"] == "doc"

    store.add_consumer(evidence_ref, "executor")
    updated, _ = store.get(evidence_ref)

    assert updated.consumers == ["executor"]
    assert len(store.list_by_trace("trace-1")) == 2


def test_state_store_can_write_large_payload_to_shared_memory(tmp_path: Path) -> None:
    store = StateStore(
        RuntimePaths(root=tmp_path),
        payload_backend="shm",
        shm_threshold_bytes=16,
    )
    try:
        ref = store.put_text(
            trace_id="trace-shm",
            producer="planner",
            text="large shared memory payload",
        )

        record, payload = store.get(ref)

        assert payload == "large shared memory payload"
        assert record.payload_ref.startswith("shm://")
        assert record.metadata["payload_backend"] == "shm"
        assert store.shm_transfer_count("trace-shm") == 1
        assert store.shm_transfer_bytes("trace-shm") == record.size_bytes
    finally:
        store.close()


def test_state_store_keeps_small_payload_file_backed_when_shm_enabled(
    tmp_path: Path,
) -> None:
    store = StateStore(
        RuntimePaths(root=tmp_path),
        payload_backend="shm",
        shm_threshold_bytes=1024,
    )
    try:
        ref = store.put_text(trace_id="trace-file", producer="planner", text="small")

        record, payload = store.get(ref)

        assert payload == "small"
        assert not record.payload_ref.startswith("shm://")
        assert "payload_backend" not in record.metadata
        assert store.shm_transfer_count("trace-file") == 0
    finally:
        store.close()


def test_state_store_raises_for_missing_state(tmp_path: Path) -> None:
    store = StateStore(RuntimePaths(root=tmp_path))

    with pytest.raises(StateNotFoundError):
        store.get("state://text/state-missing")
