# IPC + Shared Memory Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, contest-relevant systems layer that demonstrates replaceable Agent transport, shared-memory State payload transfer, and benchmark/report evidence without destabilizing the current Protocol Mode.

**Architecture:** Keep Text Mode unchanged as the plain-text baseline. Protocol Mode continues to run synchronously, but its scheduler invokes agents through an `AgentTransport` abstraction and records transport metrics. State payloads remain file-backed by default, with an opt-in Python shared-memory backend for larger Protocol Mode payloads.

**Tech Stack:** Python stdlib sockets, Python `multiprocessing.shared_memory`, Pydantic metrics, existing Rust Core optional helpers, pytest/ruff/mypy.

---

## Scope

Implement the lite version of P1, P2, and P5 from `2026-06-08-ipc-wasm-ebpf-enhancement-plan.md`.

Keep as future work:

- P3 WASM sandbox. The plan remains documented, but it is not implemented in this iteration.
- P4 eBPF observability. It remains a future optional Linux/openEuler demo path.

## Files

- Create `src/agentmesh/protocol/transport.py`: transport abstraction, in-process agent invocation transport, socket frame helper, transport metrics.
- Modify `src/agentmesh/runtime/scheduler.py`: call agents through `InProcTransport`, preserving current behavior.
- Modify `src/agentmesh/config.py`: add opt-in state payload backend config.
- Modify `src/agentmesh/state/store.py`: add file/shared-memory payload backend selection and record metadata.
- Modify `src/agentmesh/eval/metrics.py`: add transport and shared-memory counters.
- Modify `src/agentmesh/modes/protocol_mode.py`: wire config into `StateStore`, include transport/shared-memory metrics.
- Modify `src/agentmesh/eval/benchmark.py`: aggregate new metrics.
- Modify `src/agentmesh/eval/report.py`: show transport and shared-memory evidence.
- Create `tests/test_transport.py`: transport metrics and socket framing.
- Create or modify `tests/test_state_store.py`: shared-memory payload backend.
- Modify `tests/test_modes_and_benchmark.py`: report fields.

## Tasks

### Task 1: Transport Abstraction

- [ ] Add `TransportMetrics`, `AgentTransport`, `InProcTransport`, and `SocketFrameTransport`.
- [ ] Update `ProtocolScheduler` to use `InProcTransport` by default.
- [ ] Add tests proving scheduler behavior remains unchanged and socket framing round-trips AMP messages.

### Task 2: Shared-Memory State Payload Backend

- [ ] Add `StateConfig` with `AGENTMESH_STATE_PAYLOAD_BACKEND=file|shm` and `AGENTMESH_STATE_SHM_THRESHOLD_BYTES`.
- [ ] Add shared-memory writes for encoded non-blob payloads above threshold.
- [ ] Keep file backend as default and fallback if shared memory is unavailable.
- [ ] Add tests for file default, shared-memory opt-in, and metadata/counter correctness.

### Task 3: Benchmark and Report Evidence

- [ ] Add `transport_type`, transport message/byte/latency counters, shared-memory state counters to `RunMetrics`.
- [ ] Aggregate those counters in `BenchmarkSummary`.
- [ ] Add report lines for transport and state backend metrics.
- [ ] Add regression tests for benchmark/report output.

### Task 4: Verification

- [ ] Run `uv run ruff check src tests`.
- [ ] Run `uv run mypy src`.
- [ ] Run `uv run pytest`.

## Acceptance Criteria

- Text Mode remains untouched: no Rust, no StateRef, no shared memory, no transport abstraction metrics.
- Protocol Mode still produces the same dynamic routing behavior.
- Protocol Mode can opt into shared-memory State payloads via environment/config without changing public StateRef shape.
- Benchmark/report visibly show transport and shared-memory metrics.
- Full test suite passes.
