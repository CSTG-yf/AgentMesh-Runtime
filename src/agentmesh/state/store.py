import sqlite3
from pathlib import Path
from typing import Any

import orjson

from agentmesh.errors import StateNotFoundError
from agentmesh.state.refs import parse_state_ref
from agentmesh.state.schema import StateRecord, StateType
from agentmesh.storage.jsonl import append_jsonl, read_jsonl
from agentmesh.storage.paths import RuntimePaths


class StateStore:
    def __init__(self, paths: RuntimePaths) -> None:
        self.paths = paths
        self.paths.ensure()
        self._records: dict[str, StateRecord] = {}
        self._init_index()
        self._load_records()

    def put_text(
        self,
        trace_id: str,
        producer: str,
        text: str,
        parent_state_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._put(
            StateType.TEXT,
            trace_id,
            producer,
            text,
            parent_state_refs or [],
            metadata or {},
        )

    def put_embedding(
        self,
        trace_id: str,
        producer: str,
        embedding: list[float],
        parent_state_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._put(
            StateType.EMBEDDING,
            trace_id,
            producer,
            embedding,
            parent_state_refs or [],
            metadata or {"dimensions": len(embedding)},
        )

    def put_summary(
        self,
        trace_id: str,
        producer: str,
        summary: str,
        parent_state_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._put(
            StateType.SUMMARY,
            trace_id,
            producer,
            summary,
            parent_state_refs or [],
            metadata or {},
        )

    def put_evidence(
        self,
        trace_id: str,
        producer: str,
        evidence: list[dict[str, Any]],
        parent_state_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._put(
            StateType.EVIDENCE,
            trace_id,
            producer,
            evidence,
            parent_state_refs or [],
            metadata or {},
        )

    def put_code_result(
        self,
        trace_id: str,
        producer: str,
        result: dict[str, Any],
        parent_state_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._put(
            StateType.CODE_RESULT,
            trace_id,
            producer,
            result,
            parent_state_refs or [],
            metadata or {},
        )

    def put_blob(
        self,
        trace_id: str,
        producer: str,
        data: bytes,
        parent_state_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        payload_path = self.paths.state_payload_dir / f"state-blob-{len(self._records):012d}.bin"
        payload_path.write_bytes(data)
        record = StateRecord(
            trace_id=trace_id,
            state_type=StateType.BLOB,
            producer=producer,
            parent_state_refs=parent_state_refs or [],
            payload_ref=str(payload_path),
            size_bytes=len(data),
            metadata=metadata or {},
        )
        return self._record(record)

    def get(self, ref: str) -> tuple[StateRecord, Any]:
        parsed = parse_state_ref(ref)
        record = self._records.get(parsed.state_id)
        if record is None:
            raise StateNotFoundError(f"State not found: {ref}")
        payload_path = Path(record.payload_ref)
        if record.state_type == StateType.BLOB:
            return record, payload_path.read_bytes()
        return record, orjson.loads(payload_path.read_bytes())

    def exists(self, ref: str) -> bool:
        parsed = parse_state_ref(ref)
        return parsed.state_id in self._records

    def add_consumer(self, ref: str, consumer: str) -> None:
        record, _payload = self.get(ref)
        if consumer not in record.consumers:
            record.consumers.append(consumer)
        self._records[record.state_id] = record
        self._upsert_index(record)
        append_jsonl(self.paths.protocol_states, record.model_dump(mode="json"))

    def mark_written_to_memory(self, ref: str) -> None:
        record, _payload = self.get(ref)
        record.written_to_memory = True
        self._records[record.state_id] = record
        self._upsert_index(record)

    def list_by_trace(self, trace_id: str) -> list[StateRecord]:
        return [record for record in self._records.values() if record.trace_id == trace_id]

    def _put(
        self,
        state_type: StateType,
        trace_id: str,
        producer: str,
        payload: Any,
        parent_state_refs: list[str],
        metadata: dict[str, Any],
    ) -> str:
        encoded = orjson.dumps(payload)
        record = StateRecord(
            trace_id=trace_id,
            state_type=state_type,
            producer=producer,
            parent_state_refs=parent_state_refs,
            payload_ref="",
            size_bytes=len(encoded),
            metadata=metadata,
        )
        payload_path = self.paths.state_payload_dir / f"{record.state_id}.json"
        payload_path.write_bytes(encoded)
        record.payload_ref = str(payload_path)
        return self._record(record)

    def _record(self, record: StateRecord) -> str:
        self._records[record.state_id] = record
        self._upsert_index(record)
        append_jsonl(self.paths.protocol_states, record.model_dump(mode="json"))
        return record.ref

    def _load_records(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT record_json FROM state_records").fetchall()
        for row in rows:
            try:
                record = StateRecord.model_validate_json(str(row[0]))
            except Exception:
                continue
            self._records[record.state_id] = record

    def _connect(self) -> sqlite3.Connection:
        self.paths.state_index.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.paths.state_index)

    def _init_index(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_records (
                    state_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    state_type TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    payload_ref TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    written_to_memory INTEGER NOT NULL,
                    record_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(state_records)").fetchall()
            }
            if "record_json" not in columns:
                conn.execute(
                    "ALTER TABLE state_records ADD COLUMN record_json TEXT NOT NULL DEFAULT '{}'"
                )
                self._backfill_record_json(conn)

    def _backfill_record_json(self, conn: sqlite3.Connection) -> None:
        for row in read_jsonl(self.paths.protocol_states):
            try:
                record = StateRecord.model_validate(row)
            except Exception:
                continue
            conn.execute(
                "UPDATE state_records SET record_json = ? WHERE state_id = ?",
                (record.model_dump_json(), record.state_id),
            )

    def _upsert_index(self, record: StateRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO state_records (
                    state_id, trace_id, state_type, producer, payload_ref, size_bytes,
                    created_at, written_to_memory, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.state_id,
                    record.trace_id,
                    record.state_type.value,
                    record.producer,
                    record.payload_ref,
                    record.size_bytes,
                    record.created_at.isoformat(),
                    int(record.written_to_memory),
                    record.model_dump_json(),
                ),
            )
