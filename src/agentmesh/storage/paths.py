import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RuntimePaths(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def latest_run(self) -> Path:
        return self.runs_dir / "latest"

    @property
    def text_dir(self) -> Path:
        return self.latest_run / "text"

    @property
    def protocol_dir(self) -> Path:
        return self.latest_run / "protocol"

    @property
    def data_dir(self) -> Path:
        return self.latest_run / "data"

    @property
    def global_data_dir(self) -> Path:
        return self.root / "data"

    @property
    def state_payload_dir(self) -> Path:
        return self.data_dir / "states"

    @property
    def sandbox_dir(self) -> Path:
        return self.latest_run / "sandbox"

    @property
    def protocol_messages(self) -> Path:
        return self.protocol_dir / "messages.jsonl"

    @property
    def protocol_agent_io(self) -> Path:
        return self.protocol_dir / "agent_io.jsonl"

    @property
    def protocol_states(self) -> Path:
        return self.protocol_dir / "states.jsonl"

    @property
    def protocol_memory(self) -> Path:
        return self.protocol_dir / "memory.jsonl"

    @property
    def protocol_memory_hits(self) -> Path:
        return self.protocol_dir / "memory_hits.jsonl"

    @property
    def protocol_trace(self) -> Path:
        return self.protocol_dir / "trace.jsonl"

    @property
    def text_messages(self) -> Path:
        return self.text_dir / "messages.jsonl"

    @property
    def text_agent_io(self) -> Path:
        return self.text_dir / "agent_io.jsonl"

    @property
    def text_trace(self) -> Path:
        return self.text_dir / "trace.jsonl"

    @property
    def state_index(self) -> Path:
        return self.data_dir / "state_index.sqlite"

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite"

    @property
    def global_memory_db(self) -> Path:
        return self.global_data_dir / "agentmesh_memory.sqlite"

    @property
    def memory_maintenance_log(self) -> Path:
        return self.latest_run / "memory_maintenance.jsonl"

    @property
    def benchmark_dir(self) -> Path:
        return self.latest_run / "benchmarks"

    def benchmark_suite_dir(self, suite_name: str) -> Path:
        return self.benchmark_dir / _slugify_path_name(suite_name)

    @property
    def benchmark_summary(self) -> Path:
        return self.latest_run / "benchmark_summary.csv"

    def benchmark_suite_summary(self, suite_name: str) -> Path:
        return self.benchmark_suite_dir(suite_name) / "benchmark_summary.csv"

    @property
    def benchmark_detail(self) -> Path:
        return self.latest_run / "benchmark_detail.jsonl"

    def benchmark_suite_detail(self, suite_name: str) -> Path:
        return self.benchmark_suite_dir(suite_name) / "benchmark_detail.jsonl"

    @property
    def experiment_report(self) -> Path:
        return self.latest_run / "experiment_report.md"

    def benchmark_suite_report(self, suite_name: str) -> Path:
        return self.benchmark_suite_dir(suite_name) / "experiment_report.md"

    @classmethod
    def cwd(cls) -> "RuntimePaths":
        return cls(root=Path.cwd())

    def ensure(self) -> None:
        for path in [
            self.text_dir,
            self.protocol_dir,
            self.state_payload_dir,
            self.sandbox_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def _slugify_path_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "benchmark"
