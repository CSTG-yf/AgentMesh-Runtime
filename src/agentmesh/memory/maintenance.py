from __future__ import annotations

import threading
from dataclasses import dataclass

from agentmesh.memory.lifecycle import MemoryLifecyclePolicy
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.state.embedding import EmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.jsonl import append_jsonl
from agentmesh.storage.paths import RuntimePaths


@dataclass(frozen=True)
class MemoryMaintenanceConfig:
    enabled: bool = True
    interval_seconds: float = 60.0
    max_items: int = 20


class MemoryMaintenanceWorker:
    def __init__(
        self,
        *,
        paths: RuntimePaths,
        state_store: StateStore,
        encoder: EmbeddingEncoder,
        config: MemoryMaintenanceConfig | None = None,
        lifecycle: MemoryLifecyclePolicy | None = None,
    ) -> None:
        self.paths = paths
        self.store = SQLiteMemoryStore(
            paths=paths,
            state_store=state_store,
            encoder=encoder,
            db_path=paths.global_memory_db,
            log_writes=False,
        )
        self.config = config or MemoryMaintenanceConfig()
        self.lifecycle = lifecycle or MemoryLifecyclePolicy()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="agentmesh-memory-maintenance")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def run_once(self) -> dict[str, int]:
        archived = 0
        checked = 0
        for unit in self.store.active_units()[: self.config.max_items]:
            checked += 1
            decision = self.lifecycle.decide(unit)
            if decision.action == "archive":
                self.store.archive(unit.memory_id, decision.reason)
                archived += 1
                append_jsonl(
                    self.paths.memory_maintenance_log,
                    {
                        "memory_id": unit.memory_id,
                        "action": "archive",
                        "reason": decision.reason,
                        "importance_score": decision.importance_score,
                    },
                )
        return {"checked": checked, "archived": archived}

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                pass
            self._stop.wait(self.config.interval_seconds)
