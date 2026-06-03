# Shared Memory Design

MemoryUnit stores reusable task knowledge:

- memory id
- source agent
- task topic
- summary
- tags
- evidence refs
- state refs
- embedding ref
- confidence and validity score
- provenance trace id

Retrieval modes:

- SQLite FTS5 keyword search.
- In-memory tag filtering over stored tag metadata.
- HashEmbedding semantic search using cosine similarity.
