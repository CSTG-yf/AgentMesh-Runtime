import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clear_runtime_history.py"


def test_clear_runtime_history_dry_run_does_not_delete(tmp_path: Path) -> None:
    memory_db = tmp_path / "data" / "agentmesh_memory.sqlite"
    user_task = tmp_path / "runs" / "latest" / "user_tasks" / "ask.txt"
    memory_db.parent.mkdir(parents=True)
    user_task.parent.mkdir(parents=True)
    memory_db.write_text("memory", encoding="utf-8")
    user_task.write_text("hello", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dry run only" in result.stdout
    assert memory_db.exists()
    assert user_task.exists()


def test_clear_runtime_history_yes_deletes_memory_and_input_history(
    tmp_path: Path,
) -> None:
    targets = [
        tmp_path / "data" / "agentmesh_memory.sqlite",
        tmp_path / "data" / "agentmesh_memory.sqlite-wal",
        tmp_path / "runs" / "latest" / "data" / "memory.sqlite",
        tmp_path / "runs" / "latest" / "data" / "states" / "state.json",
        tmp_path / "runs" / "latest" / "user_tasks" / "ask.txt",
        tmp_path / "runs" / "latest" / "protocol" / "agent_io.jsonl",
        tmp_path / "runs" / "latest" / "text" / "agent_io.jsonl",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("history", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--yes"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Cleared AgentMesh memories" in result.stdout
    assert not any(target.exists() for target in targets)
