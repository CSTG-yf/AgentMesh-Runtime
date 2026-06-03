# Agent Message Protocol

AMP messages carry compact collaboration metadata instead of repeated natural-language context.

Required fields include:

- `message_id`
- `trace_id`
- `source_agent`
- `target_agent`
- `msg_type`
- `action`
- `params`
- `result`
- `capability`
- `state_refs`
- `created_at`

Validation rules:

- `INVOKE` requires `action`.
- `RESULT` requires non-empty `result`.
- `ERROR` requires `result.error_code` and `result.message`.
- All `state_refs` must start with `state://`.

Serialization uses orjson through `agentmesh.protocol.codec`.
