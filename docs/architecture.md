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
