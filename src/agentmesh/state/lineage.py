from agentmesh.state.schema import StateRecord


def lineage_event(record: StateRecord) -> dict[str, object]:
    return {
        "state_ref": record.ref,
        "producer": record.producer,
        "consumers": record.consumers,
        "parent_state_refs": record.parent_state_refs,
        "trace_id": record.trace_id,
        "written_to_memory": record.written_to_memory,
    }
