# Benchmark Design

The benchmark runner executes the same tasks in Text Mode and Protocol Mode.

Tracked metrics:

- message count
- text characters and estimated tokens
- protocol bytes
- state transfer count and bytes
- memory query count and hit count
- latency
- deterministic quality score

Generated artifacts:

- `runs/latest/benchmark_summary.csv`
- `runs/latest/benchmark_detail.jsonl`
- `runs/latest/experiment_report.md`

## Fairness Rules

Text Mode and Protocol Mode must run the same task suite, repeat count, deterministic agents,
and sandbox behavior. Rust optimizations may reduce serialization, indexing, state-transfer,
and retrieval overhead, but must not change the task content, memory write policy, or quality
scoring function.

For reproducible contest evaluation, benchmark runs should avoid project-local `.env` LLM
settings unless explicitly evaluating the optional LLM-backed mode.
