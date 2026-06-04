# Architecture

AgentMesh-Runtime is organized as small, testable subsystems:

- `protocol`: AMP message schema, message types, codec, and capability descriptors.
- `runtime`: agent registry, routing helpers, event bus, scheduler, and default orchestrator.
- `agents`: deterministic Planner, Retriever, Executor, and Summarizer implementations.
- `state`: StateRecord, StateRef parsing, payload storage, lineage events, and HashEmbedding.
- `memory`: SQLite-backed MemoryUnit storage with FTS, tag, and semantic search.
- `modes`: Text Mode baseline and Protocol Mode state-passing collaboration.
- `sandbox`: local subprocess Python runner with timeout and output limits.
- `eval`: benchmark metrics, suite execution, quality scoring, and report generation.

The first version intentionally uses synchronous execution and local storage. This keeps the openEuler deployment simple while preserving clean interfaces for later replacement with IPC, sockets, shared memory, or stronger sandboxing.

## Contest Comparison Model

The evaluation compares two communication models on the same task suite:

- Text Mode baseline: agents pass the full accumulated natural-language context to the next agent. Metrics record this as `communication_model="plain_text"` and count `wire_bytes` from full text payloads.
- Protocol Mode candidate: agents exchange AMP structured messages containing compact `state://...` references. Actual payloads are stored in `StateStore`, and hot-path codec, StateRef parsing, embedding, vector top-k, and sandbox subprocess primitives can use Rust Core when installed.

The benchmark summary reports both token savings and wire-byte savings. `ProtocolWireBytes` is the compact `STATE_REF` handoff payload sent between agents. `ProtocolCompactMessageBytes` is the full compact AMP message size, backed by Rust/msgpack helpers when Rust Core is installed. `ProtocolJsonWireBytes` is reported separately for readable JSON logs. `protocol_state_payload_bytes` is also separate because State payloads are persisted shared state, not repeatedly transmitted through the agent bus.

## Rust Core Optimization

The runtime keeps Python for agent policies, CLI, prompts, LLM integration, and reports.
Hot-path system primitives are exposed through an optional `agentmesh_core` Rust extension:

- AMP codec and StateRef parsing.
- HashEmbedding and cosine similarity.
- Cosine top-k semantic search for memory retrieval.
- Fast StateStore reload from the SQLite index instead of repeated JSONL replay.
- JSON and msgpack codec helpers for protocol payload experiments.
- Sandbox subprocess management used by `SandboxRunner` when Rust Core is installed.
- Warm Python worker reuse in `SandboxRunner` to avoid repeated interpreter cold starts.

This preserves reproducible Python-level behavior while reducing runtime overhead in Protocol Mode.

## LLM-backed Agent Handoff

The default agents remain deterministic when no `.env` LLM configuration is present. When `AGENTMESH_LLM_BASE_URL`, `AGENTMESH_LLM_API_KEY`, and `AGENTMESH_LLM_MODEL` are configured, the same Protocol Mode path becomes model-backed:

- `PlannerAgent` calls the model and writes its response into the planner `SummaryState`.
- `RetrieverAgent` calls the model and inserts model evidence into the `EvidenceState`.
- `ExecutorAgent` calls the model for validation and stores that validation in the `CodeResultState`.
- `SummarizerAgent` calls the model and its response becomes the final answer, final `SummaryState`, and persisted `MemoryUnit` summary.

This keeps the contest comparison intact: Text Mode still represents long-context language passing, while Protocol Mode transfers compact StateRefs whose payloads can be deterministic or LLM-generated.
