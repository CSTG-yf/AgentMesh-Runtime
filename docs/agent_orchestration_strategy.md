# Agent Orchestration Strategy

This strategy maps common user goals to AgentMesh Protocol Mode routes. It applies
four development patterns without adding runtime dependencies:

- LangGraph: model the flow as explicit state transitions.
- AutoGen: use a supervisor-style planner to choose specialist agents.
- Mem0: keep memory access scoped to requests that need history or preferences.
- OpenHands: treat code, repository, sandbox, and command execution as
  security-sensitive work.

## Route Matrix

| User goal | Business intent | Route | Retrieval | Executor | Memory policy |
| --- | --- | --- | --- | --- | --- |
| Greeting, identity, casual chat | `conversation` | planner -> summarizer | No | No | Do not retrieve old memories by default |
| User preferences, remember/forget/history | `memory` | planner -> retriever -> summarizer | Yes | No | Retrieve or update scoped memory only |
| Factual lookup, docs, sources, citations | `retrieval` | planner -> retriever -> summarizer | Yes | No | Use retrieved evidence as answer grounding |
| Analysis, architecture, report, comparison | `analysis` | planner -> retriever -> summarizer | Yes | No | Persist reusable summary when useful |
| Code implementation, bug fix, test, debug | `validation` | planner -> retriever -> executor -> summarizer | Yes | Yes | Use repo/evidence context before execution |
| Benchmark, validation, metric computation | `validation` | planner -> retriever -> executor -> summarizer | Yes | Yes | Store validated claims and evidence gaps |
| Pure calculation over user-provided data | `validation` | planner -> executor -> summarizer | No | Yes | Avoid unrelated memory retrieval |
| Code review/security/performance review | `review` | planner -> retriever -> summarizer | Yes | No | Review findings should be evidence-led |
| Summary of provided context only | `summarization` | planner -> summarizer | No | No | Do not retrieve unless requested |
| Empty or unclear input | `clarification` | planner -> summarizer | No | No | Ask for the missing goal or artifact |

## Policy Notes

The planner is the supervisor. It classifies the user goal first, then selects
capabilities. Capabilities determine the actual route, so future agents can
replace the default planner, retriever, executor, or summarizer without changing
business policy.

Memory is opt-in by intent. Casual chat should not pull arbitrary historical
memory, because stale memories can override simple answers. Memory requests,
document lookup, analysis, and code work may use retrieval because history or
evidence is likely useful.

Code work uses the OpenHands-style safety boundary: repository context and
evidence first, sandbox execution second, final synthesis last. Code review does
not execute by default; it prioritizes findings, risks, and tests.

Calculation over data supplied directly by the user skips retrieval. This avoids
polluting deterministic tasks with unrelated memory hits and keeps latency low.
