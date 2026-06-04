import csv
from pathlib import Path

from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def generate_report(paths: RuntimePaths) -> Path:
    summary = _read_summary(paths.benchmark_summary)
    messages = read_jsonl(paths.protocol_messages)[:5]
    states = read_jsonl(paths.protocol_states)[:5]
    memory = read_jsonl(paths.protocol_memory)[:5]
    lines = [
        "# AgentMesh Runtime Experiment Report",
        "",
        "## Metrics",
        "",
        "Comparison target: plain text agent handoff vs structured AMP + StateRef handoff.",
        "",
        f"- TokenSavingRate: {summary.get('token_saving_rate', '0')}",
        f"- WireBytesReductionRate: {summary.get('wire_bytes_reduction_rate', '0')}",
        f"- TextWireBytes: {summary.get('text_wire_bytes', '0')}",
        f"- ProtocolWireBytes: {summary.get('protocol_wire_bytes', '0')}",
        f"- ProtocolCompactMessageBytes: {summary.get('protocol_compact_message_bytes', '0')}",
        f"- ProtocolJsonWireBytes: {summary.get('protocol_json_wire_bytes', '0')}",
        f"- ProtocolStatePayloadBytes: {summary.get('protocol_state_payload_bytes', '0')}",
        f"- LatencyReductionRate: {summary.get('latency_reduction_rate', '0')}",
        f"- MemoryHitRate: {summary.get('memory_hit_rate', '0')}",
        f"- QualityPreservationRate: {summary.get('quality_preservation_rate', '0')}",
        f"- RustCoreEnabledRuns: {summary.get('rust_core_enabled_runs', '0')}",
        f"- RustSandboxBackendRuns: {summary.get('rust_sandbox_backend_runs', '0')}",
        "",
        "## Protocol Log Sample",
        "```json",
        *[str(item) for item in messages],
        "```",
        "",
        "## StateRef Sample",
        "```json",
        *[str(item) for item in states],
        "```",
        "",
        "## MemoryUnit Sample",
        "```json",
        *[str(item) for item in memory],
        "```",
        "",
    ]
    paths.experiment_report.parent.mkdir(parents=True, exist_ok=True)
    paths.experiment_report.write_text("\n".join(lines), encoding="utf-8")
    return paths.experiment_report


def _read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}
