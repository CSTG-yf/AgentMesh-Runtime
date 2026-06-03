import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from agentmesh.errors import SandboxTimeoutError
from agentmesh.sandbox.limits import SandboxLimits


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    latency_ms: int


class SandboxRunner:
    def __init__(self, base_dir: Path, limits: SandboxLimits | None = None) -> None:
        self.base_dir = base_dir
        self.limits = limits or SandboxLimits()

    def run_python(self, code: str) -> SandboxResult:
        sandbox_dir = self.base_dir / f"run-{uuid4().hex[:8]}"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        script = sandbox_dir / "main.py"
        script.write_text(code, encoding="utf-8")
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, str(script.resolve())],
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=self.limits.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError("Sandbox execution timed out") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SandboxResult(
            stdout=completed.stdout[: self.limits.max_output_chars],
            stderr=completed.stderr[: self.limits.max_output_chars],
            exit_code=completed.returncode,
            latency_ms=latency_ms,
        )
