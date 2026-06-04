from pydantic import BaseModel, Field


class RunMetrics(BaseModel):
    message_count: int = 0
    text_chars: int = 0
    estimated_tokens: int = 0
    communication_model: str = ""
    wire_bytes: int = 0
    text_wire_bytes: int = 0
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
    memory_query_count: int = 0
    memory_hit_count: int = 0
    latency_ms: int = 0
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)
    answer_quality_score: float = 1.0

    @property
    def memory_hit_rate(self) -> float:
        if self.memory_query_count == 0:
            return 0.0
        return self.memory_hit_count / self.memory_query_count


class ModeRunResult(BaseModel):
    mode: str
    trace_id: str
    answer: str
    metrics: RunMetrics


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
