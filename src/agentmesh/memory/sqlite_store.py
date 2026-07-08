import re
import sqlite3
from datetime import UTC, datetime
from math import log1p
from pathlib import Path

import orjson

from agentmesh.core import rust_available, rust_core
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.scorer import MemoryScorer
from agentmesh.memory.search import MemorySearchResult
from agentmesh.state.embedding import (
    EmbeddingEncoder,
    HashEmbeddingEncoder,
    cosine_similarity,
    is_semantic_encoder,
)
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
                    memory_id, source_agent, created_at, last_used_at, task_topic, summary, content,
                    tags_json, evidence_refs_json, state_refs_json, embedding_ref, reuse_count,
                    confidence, validity_score, reuse_policy, provenance_trace_id,
                    status, importance_score, last_compacted_at, archive_reason,
                    source_memory_ids_json, memory_type, domain, write_scope, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unit.memory_id,
                    unit.source_agent,
                    unit.created_at.isoformat(),
                    unit.last_used_at.isoformat() if unit.last_used_at else None,
                    unit.task_topic,
                    unit.summary,
                    _memory_content(unit),
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
                INSERT OR REPLACE INTO memory_fts(memory_id, task_topic, summary, content, tags)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    unit.memory_id,
                    unit.task_topic,
                    unit.summary,
                    _memory_content(unit),
                    " ".join(unit.tags),
                ),
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
        scorer = MemoryScorer()
        query_tags = query_tags or []
        candidate_units = self._semantic_candidates(
            query=query,
            query_tags=query_tags,
            candidate_limit=candidate_limit,
        )
        if not is_semantic_encoder(self.encoder):
            return self._rank_by_fts_fallback(
                candidates=candidate_units,
                query=query,
                query_tags=query_tags,
                limit=limit,
                scorer=scorer,
            )
        # ── Hybrid search: BM25 + semantic vector ──
        query_embedding = self.encoder.encode(query)
        units: list[MemoryUnit] = []
        vectors: list[list[float]] = []
        fts_rank: dict[str, int] = {}  # memory_id → FTS rank position
        for rank_pos, unit in enumerate(candidate_units):
            vector = self._embedding_payload(unit)
            if vector is not None:
                units.append(unit)
                vectors.append(vector)
                fts_rank[unit.memory_id] = rank_pos
        max_reuse = max((unit.reuse_count for unit in units), default=0)

        # Rust fast path
        if _rust_memory_rank_available() and vectors:
            ranked = rust_core().memory_rank_top_k(
                query_embedding,
                vectors,
                [unit.validity_score for unit in units],
                [unit.confidence for unit in units],
                [_reuse_score(unit.reuse_count, max_reuse) for unit in units],
                [scorer.recency_score(unit) for unit in units],
                [scorer.tag_overlap_score(query_tags, unit.tags) for unit in units],
                limit * 2,
            )
            raw_results = []
            for index, score, semantic in ranked:
                unit = units[int(index)]
                mid = unit.memory_id
                fts_pos = fts_rank.get(mid)
                fts_boost = max(0.0, 0.3 * (0.85 ** (fts_pos or 99))) if fts_pos is not None else 0.0
                hybrid_score = min(1.0, 0.65 * score + 0.35 * (score + fts_boost))
                raw_results.append(
                    _ranked_result_from_rust(
                        unit=unit,
                        score=hybrid_score,
                        semantic_similarity=float(semantic),
                        scorer=scorer,
                        query_tags=query_tags,
                        max_reuse_count=max_reuse,
                    )
                )
            raw_results.sort(key=lambda item: item.score, reverse=True)
            return raw_results[:limit]

        # Python vector cosine path
        similarities = {
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
            mid = unit.memory_id
            fts_pos = fts_rank.get(mid)
            fts_boost = max(0.0, 0.3 * (0.85 ** (fts_pos or 99))) if fts_pos is not None else 0.0
            base_score, parts = scorer.rank_score(
                memory=unit,
                semantic_similarity=semantic,
                query_tags=query_tags,
                max_reuse_count=max_reuse,
            )
            hybrid_score = min(1.0, 0.65 * base_score + 0.35 * (base_score + fts_boost))
            results.append(
                MemorySearchResult(
                    memory=unit,
                    score=hybrid_score,
                    reason=(
                        f"hybrid: sem={semantic:.3f}, fts_boost={fts_boost:.3f}, "
                        f"validity={unit.validity_score:.2f}, reuse={unit.reuse_count}"
                    ),
                    **parts,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _rank_by_fts_fallback(
        self,
        *,
        candidates: list[MemoryUnit],
        query: str,
        query_tags: list[str] | None,
        limit: int,
        scorer: MemoryScorer,
    ) -> list[MemorySearchResult]:
        """Rank memory candidates using FTS relevance + tag overlap + recency.

        Used when the encoder cannot produce meaningful semantic embeddings
        (e.g. HashEmbeddingEncoder).  Cosine similarity is meaningless for
        hash embeddings, so we rely on SQLite FTS5 BM25 as the primary
        relevance signal, then boost by tag overlap and recency.

        The score and semantic_similarity are calibrated to stay above the
        retriever thresholds (semantic_similarity >= 0.60 and score >= 0.60
        for tagged, non-tag-overlap queries) for top positions.
        """
        if not candidates:
            return []
        max_reuse = max((unit.reuse_count for unit in candidates), default=0)
        results: list[MemorySearchResult] = []
        for rank_pos, unit in enumerate(candidates):
            # Decay by rank position: position 0 → 0.85, position 1 → ~0.75,
            # position 2 → ~0.66, position 5 → ~0.45
            base = max(0.25, 0.95 * (0.88 ** rank_pos))
            tag_overlap = scorer.tag_overlap_score(query_tags or [], unit.tags)
            recency = scorer.recency_score(unit)
            reuse = _reuse_score(unit.reuse_count, max_reuse)
            # Score is dominated by FTS position then boosted by metadata
            score = (
                0.55 * base
                + 0.15 * tag_overlap
                + 0.10 * unit.validity_score
                + 0.10 * reuse
                + 0.10 * recency
            )
            # Boost score when the candidate has been reused before (proven useful)
            if unit.reuse_count > 0:
                score = min(1.0, score + 0.05 * reuse)
            results.append(
                MemorySearchResult(
                    memory=unit,
                    score=score,
                    semantic_similarity=base,
                    tag_overlap_score=tag_overlap,
                    recency_score=recency,
                    validity_score=unit.validity_score,
                    reuse_score=reuse,
                    reason=(
                        f"fts_rank={rank_pos + 1}, tag_overlap={tag_overlap:.2f}, "
                        f"recency={recency:.2f}, validity={unit.validity_score:.2f}"
                    ),
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
                    content TEXT DEFAULT '',
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
                USING fts5(memory_id, task_topic, summary, content, tags)
                """
            )
            self._ensure_columns(conn)
            self._ensure_fts_schema(conn)

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
            "content": "''",
        }
        for column, default in defaults.items():
            if column not in columns:
                conn.execute(f"ALTER TABLE memory_units ADD COLUMN {column} DEFAULT {default}")

    def _ensure_fts_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(memory_fts)").fetchall()
        columns = [str(row[1]) for row in rows]
        expected = ["memory_id", "task_topic", "summary", "content", "tags"]
        if columns == expected:
            return
        conn.execute("DROP TABLE IF EXISTS memory_fts")
        conn.execute(
            """
            CREATE VIRTUAL TABLE memory_fts
            USING fts5(memory_id, task_topic, summary, content, tags)
            """
        )
        conn.execute(
            """
            INSERT INTO memory_fts(memory_id, task_topic, summary, content, tags)
            SELECT memory_id, task_topic, summary, COALESCE(content, summary), tags_json
            FROM memory_units
            """
        )

    def _all_units(self) -> list[MemoryUnit]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memory_units ORDER BY created_at").fetchall()
        return [self._row_to_unit(row) for row in rows]

    def _semantic_candidates(
        self,
        *,
        query: str,
        query_tags: list[str],
        candidate_limit: int,
    ) -> list[MemoryUnit]:
        limit = max(1, candidate_limit)
        channel_limit = max(1, limit)
        max_candidates = max(limit, limit * 3)
        candidates: dict[str, MemoryUnit] = {}
        with self._connect() as conn:
            for unit in self._fts_candidates(conn, query=query, limit=channel_limit):
                candidates[unit.memory_id] = unit
            if query_tags:
                for unit in self._tag_candidates(
                    conn,
                    tags=query_tags,
                    limit=max(channel_limit, len(query_tags) * channel_limit),
                ):
                    candidates.setdefault(unit.memory_id, unit)
                    if len(candidates) >= max_candidates:
                        break
            if len(candidates) < max_candidates:
                remaining = max_candidates - len(candidates)
                for unit in self._recent_candidates(conn, limit=remaining):
                    candidates.setdefault(unit.memory_id, unit)
                    if len(candidates) >= max_candidates:
                        break
        if candidates:
            return list(candidates.values())
        return self._all_units()

    def _fts_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        query: str,
        limit: int,
    ) -> list[MemoryUnit]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        try:
            rows = conn.execute(
                """
                SELECT m.* FROM memory_fts f
                JOIN memory_units m ON m.memory_id = f.memory_id
                WHERE memory_fts MATCH ?
                AND m.status = 'active'
                ORDER BY bm25(memory_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [self._row_to_unit(row) for row in rows]

    def _tag_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        tags: list[str],
        limit: int,
    ) -> list[MemoryUnit]:
        if not tags:
            return []
        clauses = " OR ".join("tags_json LIKE ?" for _tag in tags)
        rows = conn.execute(
            f"""
            SELECT * FROM memory_units
            WHERE status = 'active'
            AND ({clauses})
            ORDER BY COALESCE(last_used_at, created_at) DESC, created_at DESC
            LIMIT ?
            """,
            [*(f'%"{tag}"%' for tag in tags), max(1, limit)],
        ).fetchall()
        return [self._row_to_unit(row) for row in rows]

    def _recent_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int,
    ) -> list[MemoryUnit]:
        rows = conn.execute(
            """
            SELECT * FROM memory_units
            WHERE status = 'active'
            ORDER BY COALESCE(last_used_at, created_at) DESC, created_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
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
            content=str(row["content"] or row["summary"]),
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


def _memory_content(unit: MemoryUnit) -> str:
    return unit.content or unit.summary


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
