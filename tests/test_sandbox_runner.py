import sys
from pathlib import Path

import pytest

from agentmesh.core import rust_available, rust_core
from agentmesh.errors import SandboxTimeoutError
from agentmesh.sandbox.limits import SandboxLimits
from agentmesh.sandbox.runner import SandboxRunner

requires_rust_core = pytest.mark.skipif(
    not rust_available(),
    reason="agentmesh_core Rust extension is not installed",
)


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


@requires_rust_core
def test_rust_sandbox_subprocess_captures_result(tmp_path: Path) -> None:
    stdout, stderr, exit_code, latency_ms = rust_core().run_python_subprocess(
        sys.executable,
        "print(2 + 4)",
        str(tmp_path),
        2_000,
        8_000,
    )

    assert exit_code == 0
    assert stdout.strip() == "6"
    assert stderr == ""
    assert latency_ms >= 0


@requires_rust_core
def test_rust_sandbox_subprocess_times_out(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError):
        rust_core().run_python_subprocess(
            sys.executable,
            "import time\ntime.sleep(2)",
            str(tmp_path),
            100,
            8_000,
        )
