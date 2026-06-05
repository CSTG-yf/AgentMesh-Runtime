import atexit
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO
from uuid import uuid4

from pydantic import BaseModel, Field

from agentmesh.core import rust_available, rust_core
from agentmesh.errors import SandboxTimeoutError
from agentmesh.sandbox.limits import SandboxLimits


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    latency_ms: int
    backend: str = "python"
    backend_failures: list[str] = Field(default_factory=list)


class SandboxRunner:
    def __init__(
        self,
        base_dir: Path,
        limits: SandboxLimits | None = None,
        use_warm_worker: bool = False,
        use_rust: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.limits = limits or SandboxLimits()
        self.use_warm_worker = use_warm_worker
        self.use_rust = use_rust

    def run_python(self, code: str) -> SandboxResult:
        backend_failures: list[str] = []
        if self.use_warm_worker:
            try:
                return self._with_failures(
                    self._run_python_warm_worker(code),
                    backend_failures,
                )
            except SandboxTimeoutError:
                raise
            except Exception as exc:
                backend_failures.append(f"warm_python_unsafe: {exc}")
        if self.use_rust and rust_available():
            try:
                return self._with_failures(self._run_python_rust(code), backend_failures)
            except TimeoutError as exc:
                raise SandboxTimeoutError("Sandbox execution timed out") from exc
            except Exception as exc:
                backend_failures.append(f"rust: {exc}")
        return self._with_failures(self._run_python_subprocess(code), backend_failures)

    def _run_python_warm_worker(self, code: str) -> SandboxResult:
        worker = _get_warm_worker(self.base_dir)
        try:
            response, latency_ms = worker.run(code, self.limits.timeout_seconds)
        except TimeoutError as exc:
            _drop_warm_worker(self.base_dir)
            raise SandboxTimeoutError("Sandbox execution timed out") from exc
        stdout = str(response.get("stdout", ""))[: self.limits.max_output_chars]
        stderr = str(response.get("stderr", ""))[: self.limits.max_output_chars]
        exit_code_raw = response.get("exit_code", -1)
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code_raw if isinstance(exit_code_raw, int) else int(str(exit_code_raw)),
            latency_ms=latency_ms,
            backend="warm_python_unsafe",
        )

    def _with_failures(
        self,
        result: SandboxResult,
        backend_failures: list[str],
    ) -> SandboxResult:
        if not backend_failures:
            return result
        return result.model_copy(update={"backend_failures": list(backend_failures)})

    def _run_python_rust(self, code: str) -> SandboxResult:
        timeout_ms = max(0, int(self.limits.timeout_seconds * 1000))
        stdout, stderr, exit_code, latency_ms = rust_core().run_python_subprocess(
            sys.executable,
            code,
            str(self.base_dir),
            timeout_ms,
            self.limits.max_output_chars,
        )
        return SandboxResult(
            stdout=str(stdout),
            stderr=str(stderr),
            exit_code=int(exit_code),
            latency_ms=int(latency_ms),
            backend="rust",
        )

    def _run_python_subprocess(self, code: str) -> SandboxResult:
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
            backend="python",
        )


class _WarmPythonWorker:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("worker.py"))],
            cwd=self.base_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def run(self, code: str, timeout_seconds: float) -> tuple[dict[str, object], int]:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError("Warm sandbox worker is not running")
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Warm sandbox worker pipes are unavailable")
            start = time.perf_counter()
            self._process.stdin.write(json.dumps({"code": code}) + "\n")
            self._process.stdin.flush()
            line = self._read_stdout_line(timeout_seconds)
            latency_ms = int((time.perf_counter() - start) * 1000)
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise RuntimeError("Warm sandbox worker response must be an object")
            return parsed, latency_ms

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def _read_stdout_line(self, timeout_seconds: float) -> str:
        if self._process.stdout is None:
            raise RuntimeError("Warm sandbox worker stdout is unavailable")
        stdout: IO[str] = self._process.stdout
        result_queue: queue.Queue[str] = queue.Queue(maxsize=1)

        def read_line() -> None:
            result_queue.put(stdout.readline())

        reader = threading.Thread(target=read_line, daemon=True)
        reader.start()
        try:
            line = result_queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("Warm sandbox worker timed out") from exc
        if not line:
            raise RuntimeError("Warm sandbox worker exited without a response")
        return line


_WARM_WORKERS: dict[Path, _WarmPythonWorker] = {}


def _get_warm_worker(base_dir: Path) -> _WarmPythonWorker:
    key = base_dir.resolve()
    worker = _WARM_WORKERS.get(key)
    if worker is None:
        worker = _WarmPythonWorker(key)
        _WARM_WORKERS[key] = worker
    return worker


def _drop_warm_worker(base_dir: Path) -> None:
    key = base_dir.resolve()
    worker = _WARM_WORKERS.pop(key, None)
    if worker is not None:
        worker.close()


def _close_warm_workers() -> None:
    for worker in list(_WARM_WORKERS.values()):
        worker.close()
    _WARM_WORKERS.clear()


atexit.register(_close_warm_workers)
