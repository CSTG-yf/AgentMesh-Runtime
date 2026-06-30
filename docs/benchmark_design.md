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

- `runs/latest/benchmarks/<suite_name>/<track>/manifest.json`
- `runs/latest/benchmarks/<suite_name>/<track>/benchmark_summary.csv`
- `runs/latest/benchmarks/<suite_name>/<track>/benchmark_detail.jsonl`
- `runs/latest/benchmarks/<suite_name>/<track>/experiment_report.md`

Each benchmark suite and execution track owns its artifact directory. The
`deterministic` and `llm` tracks never overwrite each other. Legacy artifacts
stored directly under `<suite_name>/` remain readable and are labeled
`legacy` by the dashboard.

## Reproducibility Schema

Schema version `2.0` adds an immutable experiment manifest and structured
provenance to every detail row:

- `experiment_id`: stable ID derived from schema, suite content hash, track,
  repeat count, and seed.
- `suite_sha256` and `task_sha256`: exact configuration and task input hashes.
- `track`: `deterministic` or `llm`.
- `repeat_index` and `pair_index`: sample position within the experiment.
- `pair_order`: actual Text/Protocol execution order.
- Python/platform/model/Rust summaries without API keys or secrets.

The runner alternates Text-first and Protocol-first pairs using the suite
`seed`. This reduces systematic warm-cache and service-order bias while keeping
the sequence deterministic.

Summaries retain aggregate contest metrics and add sample distributions for
Text/Protocol latency and Agent I/O tokens:

- sample count
- mean
- population standard deviation
- P50
- P95
- minimum and maximum

Deterministic and LLM tracks must be analyzed separately. Results from
different schema versions, tracks, suite hashes, or environments are not
interchangeable.

Report format:

- `# AgentMesh Runtime Experiment Report`
- `## Metrics`
- `## Protocol Log Sample`
- `## Text Agent I/O Sample`
- `## Protocol Agent I/O Sample`
- `## StateRef Sample`
- `## MemoryUnit Sample`

## Fairness Rules

Text Mode and Protocol Mode must run the same task suite, repeat count, and quality scoring
function. Text Mode is deliberately unoptimized: it passes the complete text context from one
agent to the next and must not use Rust Core, typed envelopes, StateRefs, embeddings, shared
memory, sandbox execution, or hidden state processing. Rust optimizations may reduce
serialization, indexing, state-transfer, and retrieval overhead only in Protocol Mode.

Benchmark runs use project-local `.env` LLM settings by default, matching the interactive
Agent path. For reproducible offline contest checks, pass `--no-llm` or call
`run_benchmark(..., use_llm=False)`.

## Communication Accounting

Text Mode `wire_bytes` counts the full text handoff payloads passed between agents.
Protocol Mode `wire_bytes` counts the low-overhead typed envelope channel: session dictionary
bytes plus per-message envelope bytes. Large `params` and `result` values are represented by
payload ids in the envelope and counted separately as `protocol_typed_payload_bytes`; persisted
StateStore payload bytes are reported separately as `protocol_state_payload_bytes`.
