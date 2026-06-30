# AgentMesh-Runtime Progress

| Phase | Scope | Status | Verification |
|---|---|---|---|
| 0 | Repository, CLI skeleton, errors, storage helpers | Completed | `uv run pytest` |
| 1 | AMP schema, validation, orjson codec | Completed | `tests/test_protocol_schema.py` |
| 2 | Agent registry, capabilities, HELLO/CAPABILITY_ADVERTISE | Completed | `tests/test_capability_handshake.py` |
| 3 | StateStore, StateRef, lineage logging | Completed | `tests/test_state_store.py` |
| 4 | HashEmbedding and embedding state refs | Completed | `tests/test_embedding_ref.py` |
| 5 | Shared memory and dual runtime modes | Completed | `tests/test_memory_store.py`, `tests/test_modes_and_benchmark.py` |
| 6 | Benchmark runner and report generation | Completed | `tests/test_modes_and_benchmark.py` |
| 7 | openEuler scripts and delivery docs | Completed | Equivalent commands verified; local Windows shell has no `make` |
| 8 | Plan coverage audit and optional `.env` LLM config | Completed | `docs/implementation_audit.md`, `tests/test_config.py` |
| 9 | Optional Rust Core acceleration, msgpack helpers, sandbox subprocess backend, warm worker reuse | Completed | `tests/test_rust_*`, `tests/test_sandbox_runner.py` |
| 10 | LLM-backed real-agent state handoff in Protocol Mode | Completed | `tests/test_real_agent_protocol_mode.py` |
| 11 | Explicit plain-text vs structured AMP/Rust communication comparison metrics | Completed | `tests/test_modes_and_benchmark.py` |
| 12 | CodeAct execution, Rust typed envelope codec, prompt compare CLI, long-context benchmark, readable Chinese README | Completed | `tests/test_compare_prompt.py`, `tests/test_typed_envelope.py`, `tests/test_real_agent_protocol_mode.py` |
| 13 | Full README refresh: project overview, mechanisms, metric formulas, CLI usage, env policy, and file-by-file architecture notes | Completed | `git diff --check -- README.md` |
| 14 | Shell `/ask` routes through Protocol Mode instead of direct interactive chat | Completed | `tests/test_shell.py::test_shell_ask_routes_through_protocol_mode` |
| 15 | Shell `/ask` answer fidelity: preserve code brackets, route generic code tasks to Executor, deterministic CodeAct validation fallback, absolute sandbox paths | Completed | `tests/test_shell.py`, `tests/test_dynamic_agent_routing.py`, `tests/test_sandbox_runner.py` |
| 16 | Agent I/O audit logs for Text and Protocol modes plus scored memory reuse filtering | Completed | `tests/test_modes_and_benchmark.py`, `tests/test_shell.py`, `tests/test_hybrid_memory.py` |
| 17 | LLM-on-by-default UX for `run`, `compare`, and shell `/compare`, with `--no-llm` offline override | Completed | `tests/test_compare_prompt.py::test_prompt_compare_uses_llm_by_default_and_can_disable_it`, `tests/test_shell.py::test_shell_compare_uses_llm_by_default_and_can_disable_it` |
| 18 | Shell bare-text input routes to `/ask` Protocol Mode instead of `/compare` | Completed | `tests/test_shell.py::test_shell_parser_routes_plain_text_to_ask`, `tests/test_shell.py::test_shell_plain_text_routes_through_ask_protocol_mode` |
| 19 | Streaming compare UI: parallel Text/Protocol execution, per-agent output panels, final answers, and metrics table | Completed | `tests/test_compare_prompt.py::test_prompt_compare_streams_and_collects_agent_outputs`, `tests/test_shell.py::test_render_compare_shows_answers_agent_outputs_then_metrics` |
| 20 | Generic algorithm/code routing: algorithm and self-check requests trigger Executor CodeAct without task-specific baked-in solutions | Completed | `tests/test_dynamic_agent_routing.py::test_protocol_mode_handles_algorithm_request_without_baked_in_solution` |
| 21 | Shell `/ask` and bare-text input stream Protocol Mode agent outputs before the final answer | Completed | `tests/test_shell.py::test_shell_ask_streams_protocol_agent_outputs` |
| 22 | Review hardening: agents dereference StateRefs, RetrieverAgent owns memory search, and route normalization uses capability advertisements | Completed | `tests/test_agent_state_refs.py`, `tests/test_dynamic_agent_routing.py::test_planner_decision_normalizes_route_from_capability_advertisements`, `tests/test_hybrid_memory.py` |
| 23 | P0 engineering baseline: AgentShell regression contract, CodeAct state schema, memory ranking, static quality, and ten-round stability | Completed | `uv run ruff check .`, `uv run mypy src`, `uv run pytest` |
| 24 | P1 reproducible benchmark: track isolation, immutable manifest, paired ordering, task hashes, and distribution statistics | Completed | `tests/test_experiment_manifest.py`, `tests/test_benchmark_statistics.py`, `tests/test_modes_and_benchmark.py`, `tests/test_benchmark_dashboard.py` |

## 2026-06-30 P1 Reproducible Benchmark

- Benchmark schema `2.0` records experiment ID, suite/task SHA-256, track,
  repeat index, pair order, seed, and a secret-free environment summary.
- Deterministic and LLM artifacts use independent directories and remain
  simultaneously discoverable by the Dashboard.
- Text-first and Protocol-first execution order alternates deterministically.
- Summary output includes mean, population standard deviation, P50, P95,
  minimum, and maximum for latency and Agent I/O token samples.
- A real offline `standard` run generated six paired samples under
  `continuous_tasks/deterministic`, plus manifest, report, and Dashboard.
- Final P1 verification: Ruff passed, mypy reported no issues in 73 source
  files, pytest reported 162 passed and 13 Rust-dependent skips, and the 27
  dedicated AgentShell/compare/interactive tests passed.
- This P1 run validates experimental infrastructure, not performance claims:
  Protocol token and latency results are currently negative and remain the
  explicit optimization target for P2.

## 2026-06-30 P0 Engineering Baseline

- In-process AgentShell runs skip handshake overhead by default; formal evidence
  runs explicitly enable and record HELLO, CAPABILITY_ADVERTISE, CAPABILITY_QUERY,
  and PROTOCOL_MAP messages.
- Persisted CodeAct state preserves executable code, executor metadata, and
  generated-file write status.
- Hash embeddings use deterministic lexical-vector ranking and are not
  described as semantic embeddings.
- AgentShell bare-text routing, streaming, code fidelity, command surface,
  dashboard generation, and error recovery are covered by regression tests.
- Ten continuous Protocol Mode tasks complete with distinct trace IDs.
- Ruff passes, mypy reports no issues in 71 source files, and pytest reports
  155 passed with 13 Rust-dependent tests skipped because the extension is not
  installed in this isolated worktree.
- A real `ShellSession` smoke sequence completed `/help`, `/config`, offline
  `/compare`, bare-text Protocol Mode execution, `/dashboard`, and `/exit`.

Latest verified command:

```bash
uv sync --all-extras
uv run ruff check .
uv run mypy src
uv run pytest
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
uv run agentmesh --help
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
uv run agentmesh run --mode text --task examples/tasks/A1_requirements.txt
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
uv run agentmesh report --run runs/latest
uv run agentmesh memory search --keyword protocol
uv run agentmesh memory search --tag protocol
uv run agentmesh memory search --semantic "agent state passing"
uv run agentmesh trace show
```

Results:

- `ruff`: all checks passed.
- `mypy`: no issues in 71 source files.
- `pytest`: 155 tests passed and 13 Rust-dependent tests skipped in the isolated P0 worktree.
- Protocol demo: produced `sandbox exit 0`.
- Real LLM Protocol demo: `trace-dcefca7cdce7`, final answer came from `SummarizerAgent` model output; LLM stages dominated latency (`planner` about `19.7s`, `retriever` about `30.2s`, `executor` about `22.1s`, `summarizer` about `23.6s`).
- Benchmark: 30 logical runs, `TokenSavingRate` about `0.7055`, `WireBytesReductionRate` about `0.3300`, `TextWireBytes` `26418`, `ProtocolWireBytes` `17700`, `MemoryHitRate` about `0.9667`, `LatencyReductionRate` about `-524.29`.
- Rust Core: StateRef parsing, JSON/msgpack codec helpers, HashEmbedding, semantic top-k, Rust sandbox backend, and warm worker reuse are implemented behind Python fallback boundaries.
- Rust communication benchmark: Rust Core was available in all 30 protocol runs; `ProtocolCompactMessageBytes` was `242274` and readable `ProtocolJsonWireBytes` was `280404`. `RustSandboxBackendRuns` was `0` because `SandboxRunner` currently prefers the warm Python worker before falling back to Rust subprocess.
- Prompt compare CLI: `uv run agentmesh compare "..."` ran both modes from user input and reported token, latency, wire-byte, and memory-hit metrics.
- Long-context benchmark: 20 logical runs, `TokenSavingRate` about `0.5310`, `WireBytesReductionRate` about `0.8861`, `TextWireBytes` `128524`, `ProtocolWireBytes` `14640`, `MemoryHitRate` about `0.95`.
- Rust typed envelope codec: `uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml` succeeded; Python confirmed `encode_typed_envelope_bytes` and `decode_typed_envelope_json_text` are exported.
- Warm worker profile: repeated Protocol Mode on the same run root reduced sandbox stage from about `69ms` to about `1ms`.
- `make all`: not run in this Windows shell because `make` is not installed.
