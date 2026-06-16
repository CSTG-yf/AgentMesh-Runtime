from __future__ import annotations

import os
from collections.abc import Callable

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
        self.global_store = (
            None
            if _global_memory_disabled()
            else SQLiteMemoryStore(
                paths=paths,
                state_store=state_store,
                encoder=encoder,
                db_path=paths.global_memory_db,
                log_writes=False,
            )
        )

    def put(self, unit: MemoryUnit) -> dict[str, bool]:
        run_written = False
        global_written = False
        if self.policy.should_write(unit):
            self.run_store.put(unit.model_copy(update={"write_scope": "run"}))
            run_written = True
        if self.global_store is not None and self.policy.should_write_long_term(unit):
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
        ]
        if self.global_store is not None:
            results.extend(
                self.global_store.semantic_search_with_scores(
                    query,
                    limit=limit * 3,
                    query_tags=query_tags,
                )
            )
        deduped: dict[str, MemorySearchResult] = {}
        for result in results:
            existing = deduped.get(result.memory.memory_id)
            if existing is None or result.score > existing.score:
                deduped[result.memory.memory_id] = result
        ordered = sorted(deduped.values(), key=lambda item: item.score, reverse=True)
        return ordered[:limit]

    def keyword_search(self, keyword: str, limit: int = 5) -> list[MemoryUnit]:
        return _dedupe_units(
            _global_search_units(
                self.global_store,
                self.run_store.keyword_search(keyword, limit=limit),
                lambda store: store.keyword_search(keyword, limit=limit),
            ),
            limit=limit,
        )

    def tag_search(self, tag: str, limit: int = 5) -> list[MemoryUnit]:
        return _dedupe_units(
            _global_search_units(
                self.global_store,
                self.run_store.tag_search(tag, limit=limit),
                lambda store: store.tag_search(tag, limit=limit),
            ),
            limit=limit,
        )

    def increment_reuse(self, memory_id: str) -> None:
        self.run_store.increment_reuse(memory_id)
        if self.global_store is not None:
            self.global_store.increment_reuse(memory_id)


def _dedupe_units(units: list[MemoryUnit], limit: int) -> list[MemoryUnit]:
    deduped: dict[str, MemoryUnit] = {}
    for unit in units:
        deduped.setdefault(unit.memory_id, unit)
    return list(deduped.values())[:limit]


def _global_memory_disabled() -> bool:
    return os.getenv("AGENTMESH_DISABLE_GLOBAL_MEMORY", "").lower() in {"1", "true", "yes"}


def _global_search_units(
    global_store: SQLiteMemoryStore | None,
    run_units: list[MemoryUnit],
    search_global: Callable[[SQLiteMemoryStore], list[MemoryUnit]],
) -> list[MemoryUnit]:
    if global_store is None:
        return run_units
    return [*run_units, *search_global(global_store)]
