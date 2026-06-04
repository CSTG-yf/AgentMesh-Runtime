# Benchmark Design

The benchmark runner executes the same tasks in Text Mode and Protocol Mode.

Tracked metrics:

- message count
- text characters and estimated tokens
- typed envelope wire bytes, session dictionary bytes, and externalized payload bytes
- full compact AMP bytes and readable JSON AMP bytes
- state transfer count and bytes
- memory query count and hit count
- latency
- deterministic quality score

Generated artifacts:

- `runs/latest/benchmark_summary.csv`
- `runs/latest/benchmark_detail.jsonl`
- `runs/latest/experiment_report.md`

## Fairness Rules

Text Mode and Protocol Mode must run the same task suite, repeat count, and quality scoring
function. Text Mode is deliberately unoptimized: it passes the complete text context from one
agent to the next and must not use Rust Core, typed envelopes, StateRefs, embeddings, shared
memory, sandbox execution, or hidden state processing. Rust optimizations may reduce
serialization, indexing, state-transfer, and retrieval overhead only in Protocol Mode.

For reproducible contest evaluation, benchmark runs should avoid project-local `.env` LLM
settings unless explicitly evaluating the optional LLM-backed mode.

## Communication Accounting

Text Mode `wire_bytes` counts the full text handoff payloads passed between agents.
Protocol Mode `wire_bytes` counts the low-overhead typed envelope channel: session dictionary
bytes plus per-message envelope bytes. Large `params` and `result` values are represented by
payload ids in the envelope and counted separately as `protocol_typed_payload_bytes`; persisted
StateStore payload bytes are reported separately as `protocol_state_payload_bytes`.
