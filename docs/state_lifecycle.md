# State Lifecycle

Protocol Mode uses explicit state references:

```text
Agent produces payload
StateStore writes payload
StateStore records StateRecord
AMPMessage carries state://type/id
Downstream Agent resolves StateRef
StateStore records consumers and memory lineage
```

Supported state types are `text`, `embedding`, `summary`, `evidence`, `code_result`, `blob`, and `hidden`.
