from pydantic import BaseModel, Field

TOKEN_ESTIMATOR = "mixed_cjk"


class RunMetrics(BaseModel):
    llm_call_count: int = 0
    llm_error_count: int = 0
    message_count: int = 0
    text_chars: int = 0
    estimated_tokens: int = 0
    token_estimator: str = TOKEN_ESTIMATOR
    communication_model: str = ""
    wire_bytes: int = 0
    text_wire_bytes: int = 0
    agent_io_tokens: int = 0
    agent_io_bytes: int = 0
    per_msg_avg_tokens: float = 0.0
    protocol_total_bytes: int = 0
    structured_handoff_bytes: int = 0
    session_dictionary_bytes: int = 0
    typed_envelope_bytes: int = 0
    typed_payload_bytes: int = 0
    structured_message_bytes: int = 0
    compact_structured_message_bytes: int = 0
    protocol_bytes: int = 0
    state_transfer_count: int = 0
    state_transfer_bytes: int = 0
    rust_core_enabled: bool = False
    sandbox_backend: str = ""
    transport_type: str = ""
    transport_send_count: int = 0
    transport_bytes: int = 0
    transport_avg_latency_ms: float = 0.0
    transport_p99_latency_ms: float = 0.0
    state_shm_transfer_count: int = 0
    state_shm_transfer_bytes: int = 0
    memory_query_count: int = 0
    memory_hit_count: int = 0
    memory_query_hit_count: int = 0
    memory_reused_unit_count: int = 0
    memory_evidence_count: int = 0
    memory_evidence_bytes: int = 0
    evidence_candidate_count: int = 0
    evidence_accepted_count: int = 0
    evidence_partial_count: int = 0
    evidence_deduplicated_count: int = 0
    evidence_filtered_count: int = 0
    memory_avg_score: float = 0.0
    memory_avg_semantic_similarity: float = 0.0
    memory_avg_tag_overlap_score: float = 0.0
    latency_ms: int = 0
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)
    answer_quality_score: float | None = None
    quality_rule_id: str = "unscored"
    quality_passed: bool | None = None
    dynamic_route: list[str] = Field(default_factory=list)
    route_policy_version: str = ""
    route_reason: str = ""
    selected_agents: list[str] = Field(default_factory=list)
    skipped_agents: list[str] = Field(default_factory=list)
    feedback_round_count: int = 0
    planner_refine_count: int = 0
    retriever_refine_count: int = 0
    tool_feedback_count: int = 0
    context_audits: list[dict[str, object]] = Field(default_factory=list)
    context_original_chars: int = 0
    context_retained_chars: int = 0
    context_safe_fallback_count: int = 0

    @property
    def memory_hit_rate(self) -> float:
        if self.memory_query_count == 0:
            return 0.0
        query_hits = self.memory_query_hit_count or min(
            self.memory_hit_count,
            self.memory_query_count,
        )
        return min(1.0, query_hits / self.memory_query_count)


class ModeRunResult(BaseModel):
    mode: str
    trace_id: str
    answer: str
    metrics: RunMetrics


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if TOKEN_ESTIMATOR == "char_div_4":
        return max(1, len(text) // 4)
    cjk = sum(1 for char in text if _is_cjk_or_punctuation(char))
    ascii_other = len(text) - cjk
    return max(1, int(cjk * 0.67 + ascii_other / 4))


def _is_cjk_or_punctuation(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or "\u3000" <= char <= "\u303f"
