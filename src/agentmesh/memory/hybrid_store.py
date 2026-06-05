from __future__ import annotations

from agentmesh.memory.policy import MemoryWritePolicy
from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.search import MemorySearchResult
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.state.embedding import EmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


class HybridMemoryStore:
    def __init__(
        self,
        *,
        paths: RuntimePaths,
        state_store: StateStore,
        encoder: EmbeddingEncoder,
        policy: MemoryWritePolicy | None = None,
    ) -> None:
        self.paths = paths
        self.state_store = state_store
        self.encoder = encoder
        self.policy = policy or MemoryWritePolicy()
        self.run_store = SQLiteMemoryStore(
            paths=paths,
            state_store=state_store,
            encoder=encoder,
            db_path=paths.memory_db,
            log_writes=True,
        )
        self.global_store = SQLiteMemoryStore(
            paths=paths,
            state_store=state_store,
            encoder=encoder,
            db_path=paths.global_memory_db,
            log_writes=False,
        )

    def put(self, unit: MemoryUnit) -> dict[str, bool]:
        run_written = False
        global_written = False
        if self.policy.should_write(unit):
            self.run_store.put(unit.model_copy(update={"write_scope": "run"}))
            run_written = True
        if self.policy.should_write_long_term(unit):
            global_unit = unit.model_copy(update={"write_scope": "global"})
            self.global_store.put(global_unit)
            global_written = True
        return {"run_written": run_written, "global_written": global_written}

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        query_tags: list[str] | None = None,
    ) -> list[MemoryUnit]:
        return [
            result.memory
            for result in self.semantic_search_with_scores(
                query,
                limit=limit,
                query_tags=query_tags,
            )
        ]

    def semantic_search_with_scores(
        self,
        query: str,
        limit: int = 5,
        query_tags: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        results = [
            *self.run_store.semantic_search_with_scores(
                query,
                limit=limit * 3,
                query_tags=query_tags,
            ),
            *self.global_store.semantic_search_with_scores(
                query,
                limit=limit * 3,
                query_tags=query_tags,
            ),
        ]
        deduped: dict[str, MemorySearchResult] = {}
        for result in results:
            existing = deduped.get(result.memory.memory_id)
            if existing is None or result.score > existing.score:
                deduped[result.memory.memory_id] = result
        ordered = sorted(deduped.values(), key=lambda item: item.score, reverse=True)
        return ordered[:limit]

    def keyword_search(self, keyword: str, limit: int = 5) -> list[MemoryUnit]:
        return _dedupe_units(
            [
                *self.run_store.keyword_search(keyword, limit=limit),
                *self.global_store.keyword_search(keyword, limit=limit),
            ],
            limit=limit,
        )

    def tag_search(self, tag: str, limit: int = 5) -> list[MemoryUnit]:
        return _dedupe_units(
            [
                *self.run_store.tag_search(tag, limit=limit),
                *self.global_store.tag_search(tag, limit=limit),
            ],
            limit=limit,
        )

    def increment_reuse(self, memory_id: str) -> None:
        self.run_store.increment_reuse(memory_id)
        self.global_store.increment_reuse(memory_id)


def _dedupe_units(units: list[MemoryUnit], limit: int) -> list[MemoryUnit]:
    deduped: dict[str, MemoryUnit] = {}
    for unit in units:
        deduped.setdefault(unit.memory_id, unit)
    return list(deduped.values())[:limit]
