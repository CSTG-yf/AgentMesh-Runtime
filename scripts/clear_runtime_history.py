from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClearTarget:
    path: Path
    reason: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear AgentMesh memories and persisted user input history.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag the script only previews targets.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    targets = clear_targets(root)
    existing = [target for target in targets if target.path.exists()]

    if not existing:
        print(f"No AgentMesh memory or input history files found under {root}")
        return 0

    action = "Deleting" if args.yes else "Would delete"
    print(f"{action} {len(existing)} target(s) under {root}:")
    for target in existing:
        print(f"- {target.path}  # {target.reason}")

    if not args.yes:
        print("\nDry run only. Re-run with --yes to delete these files.")
        return 0

    for target in existing:
        remove_path(target.path)
    print("Cleared AgentMesh memories and persisted input history.")
    return 0


def clear_targets(root: Path) -> list[ClearTarget]:
    latest = root / "runs" / "latest"
    run_data = latest / "data"
    protocol = latest / "protocol"
    text = latest / "text"
    global_data = root / "data"

    targets = [
        ClearTarget(latest / "user_tasks", "persisted /ask and compare prompt files"),
        ClearTarget(run_data / "states", "StateStore payloads that may contain user input"),
        ClearTarget(run_data / "state_index.sqlite", "StateStore index"),
        ClearTarget(run_data / "memory.sqlite", "current-run memory database"),
        ClearTarget(global_data / "agentmesh_memory.sqlite", "global long-term memory database"),
        ClearTarget(protocol / "memory.jsonl", "memory write log"),
        ClearTarget(protocol / "messages.jsonl", "protocol messages with state refs/actions"),
        ClearTarget(protocol / "agent_io.jsonl", "protocol agent input/output log"),
        ClearTarget(protocol / "states.jsonl", "state lineage log"),
        ClearTarget(protocol / "trace.jsonl", "protocol trace metrics"),
        ClearTarget(text / "messages.jsonl", "text-mode message log"),
        ClearTarget(text / "agent_io.jsonl", "text-mode full-context input/output log"),
        ClearTarget(text / "trace.jsonl", "text-mode trace metrics"),
        ClearTarget(latest / "memory_maintenance.jsonl", "memory maintenance log"),
    ]
    return [*targets, *sqlite_sidecars(targets)]


def sqlite_sidecars(targets: list[ClearTarget]) -> list[ClearTarget]:
    sidecars: list[ClearTarget] = []
    for target in targets:
        if target.path.suffix != ".sqlite":
            continue
        for suffix in ("-wal", "-shm", "-journal"):
            sidecars.append(
                ClearTarget(
                    target.path.with_name(target.path.name + suffix),
                    f"SQLite sidecar for {target.path.name}",
                )
            )
    return sidecars


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
