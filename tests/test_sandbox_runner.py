from pathlib import Path

import pytest

from agentmesh.errors import SandboxTimeoutError
from agentmesh.sandbox.limits import SandboxLimits
from agentmesh.sandbox.runner import SandboxRunner


def test_sandbox_runner_captures_structured_result(tmp_path: Path) -> None:
    runner = SandboxRunner(base_dir=tmp_path, limits=SandboxLimits(timeout_seconds=2))

    result = runner.run_python("print(2 + 3)")

    assert result.exit_code == 0
    assert result.stdout.strip() == "5"
    assert result.stderr == ""
    assert result.latency_ms >= 0


def test_sandbox_runner_raises_timeout(tmp_path: Path) -> None:
    runner = SandboxRunner(base_dir=tmp_path, limits=SandboxLimits(timeout_seconds=0.1))

    with pytest.raises(SandboxTimeoutError):
        runner.run_python("import time\ntime.sleep(2)")
