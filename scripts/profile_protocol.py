from pathlib import Path
from tempfile import TemporaryDirectory

from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.storage.paths import RuntimePaths


def main() -> None:
    task = Path("examples/tasks/A1_requirements.txt")
    with TemporaryDirectory(ignore_cleanup_errors=True) as root:
        paths = RuntimePaths(root=Path(root))
        result = run_protocol_mode(task, paths)
        print(result.metrics.model_dump())


if __name__ == "__main__":
    main()
