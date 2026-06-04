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

## Rust Core Optimization

The runtime keeps Python for agent policies, CLI, prompts, LLM integration, and reports.
Hot-path system primitives are exposed through an optional `agentmesh_core` Rust extension:

- AMP codec and StateRef parsing.
- HashEmbedding and cosine similarity.
- Cosine top-k semantic search for memory retrieval.
- Fast StateStore reload from the SQLite index instead of repeated JSONL replay.

This preserves reproducible Python-level behavior while reducing runtime overhead in Protocol Mode.
