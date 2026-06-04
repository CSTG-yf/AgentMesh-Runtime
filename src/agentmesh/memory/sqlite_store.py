import sqlite3
from datetime import UTC, datetime

import orjson

from agentmesh.core import rust_available, rust_core
from agentmesh.memory.schema import MemoryUnit
from agentmesh.state.embedding import EmbeddingEncoder, HashEmbeddingEncoder, cosine_similarity
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


class SQLiteMemoryStore:
    def __init__(
        self,
        paths: RuntimePaths,
        state_store: StateStore,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.paths = paths
        self.state_store = state_store
        self.encoder = encoder or HashEmbeddingEncoder()
        self.paths.ensure()
        self._init_db()

    def put(self, unit: MemoryUnit) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_units (
                    memory_id, source_agent, created_at, last_used_at, task_topic, summary,
                    tags_json, evidence_refs_json, state_refs_json, embedding_ref, reuse_count,
                    confidence, validity_score, reuse_policy, provenance_trace_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unit.memory_id,
                    unit.source_agent,
                    unit.created_at.isoformat(),
                    unit.last_used_at.isoformat() if unit.last_used_at else None,
                    unit.task_topic,
                    unit.summary,
                    orjson.dumps(unit.tags).decode("utf-8"),
                    orjson.dumps(unit.evidence_refs).decode("utf-8"),
                    orjson.dumps(unit.state_refs).decode("utf-8"),
                    unit.embedding_ref,
                    unit.reuse_count,
                    unit.confidence,
                    unit.validity_score,
                    unit.reuse_policy,
                    unit.provenance_trace_id,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_fts(memory_id, task_topic, summary, tags)
                VALUES (?, ?, ?, ?)
                """,
                (unit.memory_id, unit.task_topic, unit.summary, " ".join(unit.tags)),
            )
        for ref in unit.state_refs + unit.evidence_refs:
            if ref.startswith("state://") and self.state_store.exists(ref):
                self.state_store.mark_written_to_memory(ref)
        append_jsonl(self.paths.protocol_memory, unit.model_dump(mode="json"))

    def keyword_search(self, keyword: str, limit: int = 5) -> list[MemoryUnit]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.* FROM memory_fts f
                JOIN memory_units m ON m.memory_id = f.memory_id
                WHERE memory_fts MATCH ?
                LIMIT ?
                """,
                (keyword, limit),
            ).fetchall()
        return [self._row_to_unit(row) for row in rows]

    def tag_search(self, tag: str, limit: int = 5) -> list[MemoryUnit]:
        units = self._all_units()
        return [unit for unit in units if tag in unit.tags][:limit]

    def semantic_search(self, query: str, limit: int = 5) -> list[MemoryUnit]:
        query_embedding = self.encoder.encode(query)
        units: list[MemoryUnit] = []
        vectors: list[list[float]] = []
        scored: list[tuple[float, MemoryUnit]] = []
        for unit in self._all_units():
            if not unit.embedding_ref:
                continue
            try:
                _record, payload = self.state_store.get(unit.embedding_ref)
            except Exception:
                continue
            if isinstance(payload, list):
                vector = [float(value) for value in payload]
                units.append(unit)
                vectors.append(vector)
                scored.append((cosine_similarity(query_embedding, payload), unit))
        if rust_available() and vectors:
            indexes = rust_core().top_k_cosine(query_embedding, vectors, limit)
            return [units[int(index)] for index in indexes]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [unit for _score, unit in scored[:limit]]

    def increment_reuse(self, memory_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_units
                SET reuse_count = reuse_count + 1, last_used_at = ?
                WHERE memory_id = ?
                """,
                (now, memory_id),
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_units (
                    memory_id TEXT PRIMARY KEY,
                    source_agent TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    task_topic TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    state_refs_json TEXT NOT NULL,
                    embedding_ref TEXT,
                    reuse_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    validity_score REAL DEFAULT 0,
                    reuse_policy TEXT DEFAULT 'verify',
                    provenance_trace_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(memory_id, task_topic, summary, tags)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        self.paths.memory_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.paths.memory_db)
        conn.row_factory = sqlite3.Row
        return conn

    def _all_units(self) -> list[MemoryUnit]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memory_units ORDER BY created_at").fetchall()
        return [self._row_to_unit(row) for row in rows]

    def _row_to_unit(self, row: sqlite3.Row) -> MemoryUnit:
        return MemoryUnit(
            memory_id=str(row["memory_id"]),
            source_agent=str(row["source_agent"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            last_used_at=datetime.fromisoformat(str(row["last_used_at"]))
            if row["last_used_at"]
            else None,
            task_topic=str(row["task_topic"]),
            summary=str(row["summary"]),
            tags=list(orjson.loads(row["tags_json"])),
            evidence_refs=list(orjson.loads(row["evidence_refs_json"])),
            state_refs=list(orjson.loads(row["state_refs_json"])),
            embedding_ref=row["embedding_ref"],
            reuse_count=int(row["reuse_count"]),
            confidence=float(row["confidence"]),
            validity_score=float(row["validity_score"]),
            reuse_policy=str(row["reuse_policy"]),
            provenance_trace_id=str(row["provenance_trace_id"]),
        )
