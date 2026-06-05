import sqlite3
from datetime import UTC, datetime
from math import log1p
from pathlib import Path
import re

import orjson

from agentmesh.core import rust_available, rust_core
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.scorer import MemoryScorer
from agentmesh.memory.search import MemorySearchResult
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
        db_path: Path | None = None,
        log_writes: bool = True,
    ) -> None:
        self.paths = paths
        self.state_store = state_store
        self.encoder = encoder or HashEmbeddingEncoder()
        self.db_path = db_path or paths.memory_db
        self.log_writes = log_writes
        self.paths.ensure()
        self._init_db()

    def put(self, unit: MemoryUnit) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_units (
                    memory_id, source_agent, created_at, last_used_at, task_topic, summary,
                    tags_json, evidence_refs_json, state_refs_json, embedding_ref, reuse_count,
                    confidence, validity_score, reuse_policy, provenance_trace_id,
                    status, importance_score, last_compacted_at, archive_reason,
                    source_memory_ids_json, memory_type, domain, write_scope, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    unit.status,
                    unit.importance_score,
                    unit.last_compacted_at.isoformat() if unit.last_compacted_at else None,
                    unit.archive_reason,
                    orjson.dumps(unit.source_memory_ids).decode("utf-8"),
                    unit.memory_type,
                    unit.domain,
                    unit.write_scope,
                    orjson.dumps(self._embedding_for_unit(unit)).decode("utf-8"),
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
        if self.log_writes:
            append_jsonl(self.paths.protocol_memory, unit.model_dump(mode="json"))

    def keyword_search(self, keyword: str, limit: int = 5) -> list[MemoryUnit]:
        query = _fts_query(keyword)
        if not query:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.* FROM memory_fts f
                JOIN memory_units m ON m.memory_id = f.memory_id
                WHERE memory_fts MATCH ?
                AND m.status = 'active'
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [self._row_to_unit(row) for row in rows]

    def tag_search(self, tag: str, limit: int = 5) -> list[MemoryUnit]:
        units = self._all_units()
        return [unit for unit in units if unit.status == "active" and tag in unit.tags][:limit]

    def semantic_search(self, query: str, limit: int = 5) -> list[MemoryUnit]:
        return [result.memory for result in self.semantic_search_with_scores(query, limit=limit)]

    def semantic_search_with_scores(
        self,
        query: str,
        limit: int = 5,
        query_tags: list[str] | None = None,
        candidate_limit: int = 50,
    ) -> list[MemorySearchResult]:
        query_embedding = self.encoder.encode(query)
        units: list[MemoryUnit] = []
        vectors: list[list[float]] = []
        for unit in self._all_units():
            if unit.status != "active":
                continue
            if not unit.embedding_ref:
                continue
            vector = self._embedding_payload(unit)
            if vector is not None:
                units.append(unit)
                vectors.append(vector)
        scorer = MemoryScorer()
        max_reuse = max((unit.reuse_count for unit in units), default=0)
        if _rust_memory_rank_available() and vectors:
            ranked = rust_core().memory_rank_top_k(
                query_embedding,
                vectors,
                [unit.validity_score for unit in units],
                [unit.confidence for unit in units],
                [_reuse_score(unit.reuse_count, max_reuse) for unit in units],
                [scorer.recency_score(unit) for unit in units],
                [scorer.tag_overlap_score(query_tags or [], unit.tags) for unit in units],
                limit,
            )
            return [
                _ranked_result_from_rust(
                    unit=units[int(index)],
                    score=float(score),
                    semantic_similarity=float(semantic),
                    scorer=scorer,
                    query_tags=query_tags,
                    max_reuse_count=max_reuse,
                )
                for index, score, semantic in ranked
            ]
        similarities: dict[str, float] = {
            unit.memory_id: cosine_similarity(query_embedding, vector)
            for unit, vector in zip(units, vectors, strict=True)
        }
        if rust_available() and hasattr(rust_core(), "top_k_cosine") and vectors:
            indexes = rust_core().top_k_cosine(
                query_embedding,
                vectors,
                min(candidate_limit, len(vectors)),
            )
            candidates = [units[int(index)] for index in indexes]
        else:
            candidates = sorted(
                units,
                key=lambda unit: similarities.get(unit.memory_id, 0.0),
                reverse=True,
            )[:candidate_limit]
        results: list[MemorySearchResult] = []
        for unit in candidates:
            semantic = similarities.get(unit.memory_id, 0.0)
            score, parts = scorer.rank_score(
                memory=unit,
                semantic_similarity=semantic,
                query_tags=query_tags,
                max_reuse_count=max_reuse,
            )
            results.append(
                MemorySearchResult(
                    memory=unit,
                    score=score,
                    reason=(
                        f"semantic={semantic:.3f}, validity={unit.validity_score:.2f}, "
                        f"reuse={unit.reuse_count}"
                    ),
                    **parts,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

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

    def archive(self, memory_id: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_units
                SET status = 'archived', archive_reason = ?
                WHERE memory_id = ?
                """,
                (reason, memory_id),
            )

    def active_units(self) -> list[MemoryUnit]:
        return [unit for unit in self._all_units() if unit.status == "active"]

    def all_units(self) -> list[MemoryUnit]:
        return self._all_units()

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
                    provenance_trace_id TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    importance_score REAL DEFAULT 0,
                    last_compacted_at TEXT,
                    archive_reason TEXT,
                    source_memory_ids_json TEXT DEFAULT '[]',
                    memory_type TEXT DEFAULT 'task_summary',
                    domain TEXT DEFAULT 'general',
                    write_scope TEXT DEFAULT 'run',
                    embedding_json TEXT DEFAULT 'null'
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(memory_id, task_topic, summary, tags)
                """
            )
            self._ensure_columns(conn)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(memory_units)").fetchall()
        }
        defaults = {
            "status": "'active'",
            "importance_score": "0",
            "last_compacted_at": "NULL",
            "archive_reason": "NULL",
            "source_memory_ids_json": "'[]'",
            "memory_type": "'task_summary'",
            "domain": "'general'",
            "write_scope": "'run'",
            "embedding_json": "'null'",
        }
        for column, default in defaults.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE memory_units ADD COLUMN {column} DEFAULT {default}")

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
            embedding_vector=_loads_embedding(row["embedding_json"]),
            reuse_count=int(row["reuse_count"]),
            confidence=float(row["confidence"]),
            validity_score=float(row["validity_score"]),
            reuse_policy=str(row["reuse_policy"]),
            provenance_trace_id=str(row["provenance_trace_id"]),
            status=str(row["status"] or "active"),
            importance_score=float(row["importance_score"] or 0.0),
            last_compacted_at=datetime.fromisoformat(str(row["last_compacted_at"]))
            if row["last_compacted_at"]
            else None,
            archive_reason=row["archive_reason"],
            source_memory_ids=list(orjson.loads(row["source_memory_ids_json"] or "[]")),
            memory_type=str(row["memory_type"] or "task_summary"),
            domain=str(row["domain"] or "general"),
            write_scope=str(row["write_scope"] or "run"),
        )

    def _embedding_for_unit(self, unit: MemoryUnit) -> list[float] | None:
        if unit.embedding_vector is not None:
            return unit.embedding_vector
        return self._embedding_payload(unit)

    def _embedding_payload(self, unit: MemoryUnit) -> list[float] | None:
        if unit.embedding_vector is not None:
            return unit.embedding_vector
        if not unit.embedding_ref:
            return None
        try:
            _record, payload = self.state_store.get(unit.embedding_ref)
        except Exception:
            return None
        if isinstance(payload, list):
            return [float(value) for value in payload]
        return None


def _loads_embedding(raw: object) -> list[float] | None:
    if not raw:
        return None
    try:
        loaded = orjson.loads(str(raw))
    except Exception:
        return None
    if not isinstance(loaded, list):
        return None
    return [float(value) for value in loaded]


def _fts_query(keyword: str) -> str:
    tokens = re.findall(r"[\w]+", keyword, flags=re.UNICODE)
    return " ".join(f'"{token}"' for token in tokens)


def _rust_memory_rank_available() -> bool:
    return rust_available() and hasattr(rust_core(), "memory_rank_top_k")


def _reuse_score(reuse_count: int, max_reuse_count: int) -> float:
    denominator = log1p(max_reuse_count)
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, log1p(reuse_count) / denominator))


def _ranked_result_from_rust(
    *,
    unit: MemoryUnit,
    score: float,
    semantic_similarity: float,
    scorer: MemoryScorer,
    query_tags: list[str] | None,
    max_reuse_count: int,
) -> MemorySearchResult:
    return MemorySearchResult(
        memory=unit,
        score=score,
        semantic_similarity=semantic_similarity,
        validity_score=unit.validity_score,
        confidence_score=unit.confidence,
        reuse_score=_reuse_score(unit.reuse_count, max_reuse_count),
        recency_score=scorer.recency_score(unit),
        tag_overlap_score=scorer.tag_overlap_score(query_tags or [], unit.tags),
        reason=(
            f"rust_rank semantic={semantic_similarity:.3f}, "
            f"validity={unit.validity_score:.2f}, reuse={unit.reuse_count}"
        ),
    )
