import csv
from pathlib import Path

from agentmesh.storage.jsonl import read_jsonl
from agentmesh.storage.paths import RuntimePaths


def generate_report(
    paths: RuntimePaths,
    *,
    suite_name: str | None = None,
    track: str | None = None,
) -> Path:
    summary_path = (
        paths.benchmark_suite_summary(suite_name, track=track)
        if suite_name is not None
        else paths.benchmark_summary
    )
    report_path = (
        paths.benchmark_suite_report(suite_name, track=track)
        if suite_name is not None
        else paths.experiment_report
    )
    summary = _read_summary(summary_path)
    messages = read_jsonl(paths.protocol_messages)[:5]
    text_agent_io = read_jsonl(paths.text_agent_io)[:5]
    protocol_agent_io = read_jsonl(paths.protocol_agent_io)[:5]
    states = read_jsonl(paths.protocol_states)[:5]
    memory = read_jsonl(paths.protocol_memory)[:5]
    trace = read_jsonl(paths.protocol_trace)
    latest_metrics = _latest_metrics(trace)
    lines = [
        "# AgentMesh Runtime Experiment Report",
        "",
        "## Metrics",
        "",
        "Comparison target: plain text agent handoff vs structured AMP + StateRef handoff.",
        "",
        f"- TokenEstimator: {summary.get('token_estimator', 'mixed_cjk')}",
        f"- TokenSavingRate: {summary.get('token_saving_rate', '0')}",
        f"- WireBytesReductionRate: {summary.get('wire_bytes_reduction_rate', '0')}",
        f"- FairWireReductionRate: {summary.get('fair_wire_reduction_rate', '0')}",
        "- AgentIoBytesReductionRate: "
        f"{summary.get('agent_io_bytes_reduction_rate', '0')}",
        f"- TextAgentIoBytes: {summary.get('text_agent_io_bytes', '0')}",
        f"- ProtocolAgentIoBytes: {summary.get('protocol_agent_io_bytes', '0')}",
        f"- TextWireBytes: {summary.get('text_wire_bytes', '0')}",
        f"- ProtocolWireBytes: {summary.get('protocol_wire_bytes', '0')}",
        "- ProtocolSessionDictionaryBytes: "
        f"{summary.get('protocol_session_dictionary_bytes', '0')}",
        f"- ProtocolTypedEnvelopeBytes: {summary.get('protocol_typed_envelope_bytes', '0')}",
        f"- ProtocolTypedPayloadBytes: {summary.get('protocol_typed_payload_bytes', '0')}",
        f"- ProtocolCompactMessageBytes: {summary.get('protocol_compact_message_bytes', '0')}",
        f"- ProtocolJsonWireBytes: {summary.get('protocol_json_wire_bytes', '0')}",
        f"- ProtocolStatePayloadBytes: {summary.get('protocol_state_payload_bytes', '0')}",
        f"- LatencyReductionRate: {summary.get('latency_reduction_rate', '0')}",
        f"- MemoryHitRate: {summary.get('memory_hit_rate', '0')}",
        f"- MemoryReusedUnitCount: {summary.get('memory_reused_unit_count', '0')}",
        f"- MemoryEvidenceCount: {summary.get('memory_evidence_count', '0')}",
        f"- MemoryEvidenceBytes: {summary.get('memory_evidence_bytes', '0')}",
        "- MemoryAvgEvidenceBytesPerQuery: "
        f"{summary.get('memory_avg_evidence_bytes_per_query', '0')}",
        "- MemoryAvgReusedUnitsPerQuery: "
        f"{summary.get('memory_avg_reused_units_per_query', '0')}",
        f"- MemoryAvgScore: {summary.get('memory_avg_score', '0')}",
        "- MemoryAvgSemanticSimilarity: "
        f"{summary.get('memory_avg_semantic_similarity', '0')}",
        "- MemoryAvgTagOverlapScore: "
        f"{summary.get('memory_avg_tag_overlap_score', '0')}",
        f"- QualityPreservationRate: {summary.get('quality_preservation_rate', '0')}",
        f"- RustCoreEnabledRuns: {summary.get('rust_core_enabled_runs', '0')}",
        f"- RustSandboxBackendRuns: {summary.get('rust_sandbox_backend_runs', '0')}",
        f"- TransportSendCount: {summary.get('transport_send_count', '0')}",
        f"- TransportBytes: {summary.get('transport_bytes', '0')}",
        f"- TransportAvgLatencyMs: {summary.get('transport_avg_latency_ms', '0')}",
        f"- TransportP99LatencyMs: {summary.get('transport_p99_latency_ms', '0')}",
        f"- StateShmTransferCount: {summary.get('state_shm_transfer_count', '0')}",
        f"- StateShmTransferBytes: {summary.get('state_shm_transfer_bytes', '0')}",
        f"- FeedbackRoundCount: {summary.get('feedback_round_count', '0')}",
        f"- PlannerRefineCount: {summary.get('planner_refine_count', '0')}",
        f"- RetrieverRefineCount: {summary.get('retriever_refine_count', '0')}",
        f"- ToolFeedbackCount: {summary.get('tool_feedback_count', '0')}",
        f"- LatestDynamicRoute: {latest_metrics.get('dynamic_route', [])}",
        f"- LatestSelectedAgents: {latest_metrics.get('selected_agents', [])}",
        f"- LatestSkippedAgents: {latest_metrics.get('skipped_agents', [])}",
        "",
        "## Protocol Log Sample",
        "```json",
        *[str(item) for item in messages],
        "```",
        "",
        "## Text Agent I/O Sample",
        "```json",
        *[str(item) for item in text_agent_io],
        "```",
        "",
        "## Protocol Agent I/O Sample",
        "```json",
        *[str(item) for item in protocol_agent_io],
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}


def _latest_metrics(trace: list[dict[str, object]]) -> dict[str, object]:
    if not trace:
        return {}
    metrics = trace[-1].get("metrics")
    return metrics if isinstance(metrics, dict) else {}
