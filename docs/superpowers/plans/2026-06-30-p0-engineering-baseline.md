# P0 Engineering Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a fully passing engineering baseline while preserving the current AgentShell experience and making protocol, CodeAct state, and memory-ranking contracts internally consistent.

**Architecture:** Keep the current runtime and shell boundaries unchanged. Repair contract drift at the narrowest ownership points: protocol configuration owns handshake defaults, protocol-mode compaction owns persisted CodeAct schema, and the memory store owns lexical-vector versus semantic-vector ranking. AgentShell receives regression tests first and is not refactored during P0.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, Rich, SQLite FTS5, PyO3/Rust Core, pytest, ruff, mypy

---

## File Map

- Modify `docs/superpowers/specs/2026-06-30-contest-scoring-hardening-design.md`
  to make AgentShell experience preservation an explicit acceptance condition.
- Modify `tests/test_shell.py` to lock the existing AgentShell interaction contract.
- Modify `src/agentmesh/config.py` and `tests/test_config.py` to restore protocol
  handshake observability by default.
- Modify `src/agentmesh/modes/protocol_mode.py`,
  `tests/test_dynamic_agent_routing.py`, and
  `tests/test_real_agent_protocol_mode.py` to stabilize the persisted CodeAct
  state schema.
- Modify `src/agentmesh/state/embedding.py`,
  `src/agentmesh/memory/sqlite_store.py`,
  `tests/test_memory_store.py`, and
  `tests/test_memory_rust_vector_index.py` to use deterministic vector ranking
  for hash embeddings without claiming they are semantic embeddings.
- Modify `docs/progress.md` only after all verification commands pass.

### Task 1: Lock the AgentShell Experience Contract

**Files:**
- Modify: `tests/test_shell.py`
- Test: `tests/test_shell.py`

- [ ] **Step 1: Add a regression test for the command surface**

Add this test beside the existing parser tests:

```python
def test_agentshell_core_command_surface_remains_available(tmp_path) -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    session = ShellSession(paths=RuntimePaths(root=tmp_path), console=console)

    assert session.handle_line("/help")
    assert session.handle_line("/config")
    assert session.handle_line("/dashboard")
    assert session.handle_line("/unknown")
    assert session.handle_line("/exit") is False

    rendered = output.getvalue()
    for command in [
        "/ask",
        "/compare",
        "/run",
        "/benchmark",
        "/memory",
        "/trace",
        "/report",
        "/dashboard",
        "/config",
        "/exit",
    ]:
        assert command in rendered
    assert "Unknown command" in rendered
```

- [ ] **Step 2: Add a regression assertion for bare-text and answer fidelity**

Extend the existing bare-text `/ask` tests rather than creating a second fake
runtime. Assert that:

```python
assert parse_shell_line("print(items[0])").name == "ask"
assert "print(items[0])" in rendered
assert "[0]" in rendered
```

The fake protocol result must contain the literal answer
`"print(items[0])\nresult = {'ok': True}"`, and the assertion must use the raw
captured console output so Rich markup stripping regressions are detected.

- [ ] **Step 3: Run the focused shell suite**

Run:

```bash
uv run pytest tests/test_shell.py tests/test_compare_prompt.py tests/test_interactive_agent.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit the AgentShell contract**

```bash
git add tests/test_shell.py docs/superpowers/specs/2026-06-30-contest-scoring-hardening-design.md
git commit -m "test(shell): lock agentshell experience contract"
```

### Task 2: Make the In-Process Handshake Policy Explicit

**Files:**
- Modify: `src/agentmesh/config.py:53-59`
- Modify: `src/agentmesh/config.py:105-108`
- Modify: `src/agentmesh/config.py:176-182`
- Modify: `tests/test_config.py`
- Test: `tests/test_capability_handshake.py`
- Test: `tests/test_protocol_schema.py`

- [ ] **Step 1: Add configuration tests before changing the default**

Add tests that make the intended behavior explicit:

```python
def test_protocol_handshake_is_skipped_for_inproc_by_default() -> None:
    config = AgentMeshConfig.from_mapping({})

    assert config.protocol.skip_handshake_for_inproc is True


def test_protocol_handshake_can_be_enabled_explicitly() -> None:
    config = AgentMeshConfig.from_mapping(
        {"AGENTMESH_PROTOCOL_SKIP_HANDSHAKE_FOR_INPROC": "false"}
    )

    assert config.protocol.skip_handshake_for_inproc is False
```

- [ ] **Step 2: Run the new configuration tests and observe the failure**

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected: both policy tests pass and document the existing lightweight default.

- [ ] **Step 3: Change the protocol default in both construction paths**

Change `ProtocolConfig` and environment parsing to:

```python
class ProtocolConfig(BaseModel):
    skip_handshake_for_inproc: bool = True
    state_summary_max_chars: int = 800
    evidence_snippet_max_chars: int = 160
    agent_log_output_max_chars: int = 1200
    agent_log_param_max_chars: int = 500
    memory_reuse_default_limit: int = 1
```

and:

```python
skip_handshake_for_inproc=_parse_bool(skip_handshake_raw, default=True),
```

For the handshake evidence integration test, write
`AGENTMESH_PROTOCOL_SKIP_HANDSHAKE_FOR_INPROC=false` to the test project
`.env`. Handshake messages remain local structured events: they must not
trigger an LLM call or be printed as extra AgentShell answer panels.

- [ ] **Step 4: Run protocol and AgentShell regression tests**

Run:

```bash
uv run pytest tests/test_config.py tests/test_capability_handshake.py tests/test_shell.py -q
```

Expected: all selected tests pass; the shell output assertions remain
unchanged.

- [ ] **Step 5: Commit the handshake fix**

```bash
git add src/agentmesh/config.py tests/test_config.py
git commit -m "fix(protocol): record capability handshake by default"
```

### Task 3: Stabilize the Persisted CodeAct Result Schema

**Files:**
- Modify: `src/agentmesh/modes/protocol_mode.py:892-927`
- Modify: `tests/test_dynamic_agent_routing.py`
- Modify: `tests/test_real_agent_protocol_mode.py`
- Test: `tests/test_sandbox_runner.py`

- [ ] **Step 1: Add direct compaction-contract tests**

Add a test for `_compact_code_result_payload` that uses this payload:

```python
payload = {
    "stdout": '{"status": "validated"}',
    "stderr": "",
    "exit_code": 0,
    "latency_ms": 3,
    "backend": "rust",
    "executor_result": {
        "validated": True,
        "llm_generated_code": True,
        "codeact_code": "print('ok')",
    },
    "generated_files": [
        {
            "path": "C:/workspace/generated_code.py",
            "relative_path": "generated_code.py",
            "written": True,
            "overwritten": False,
            "bytes": 11,
        }
    ],
    "tool_feedback": {"status": "success"},
    "codeact": {
        "code": "print('ok')",
        "generated_by_llm": True,
        "stdout": "ok",
        "stderr": "",
        "exit_code": 0,
    },
}
```

Assert:

```python
assert compacted["codeact"]["code"] == "print('ok')"
assert compacted["executor_result"]["llm_generated_code"] is True
assert compacted["generated_files"][0]["written"] is True
assert compacted["generated_files"][0]["relative_path"] == "generated_code.py"
```

- [ ] **Step 2: Run the CodeAct tests and observe schema failures**

Run:

```bash
uv run pytest tests/test_dynamic_agent_routing.py tests/test_real_agent_protocol_mode.py -q
```

Expected: failures show missing `codeact.code`, `executor_result`, and
`generated_files[].written`.

- [ ] **Step 3: Preserve bounded reconstructable CodeAct fields**

Replace the lossy compaction body with a schema-preserving bounded form:

```python
def _compact_code_result_payload(
    payload: dict[str, Any],
    *,
    text_limit: int,
) -> dict[str, Any]:
    codeact = payload.get("codeact")
    codeact_summary: dict[str, Any] = {}
    if isinstance(codeact, dict):
        code = str(codeact.get("code", ""))
        codeact_summary = {
            "code": code,
            "code_preview": _compact_text(code, text_limit),
            "code_chars": len(code),
            "generated_by_llm": bool(codeact.get("generated_by_llm")),
            "stdout": _compact_text(str(codeact.get("stdout", "")), text_limit),
            "stderr": _compact_text(
                str(codeact.get("stderr", "")),
                max(120, text_limit // 2),
            ),
            "exit_code": codeact.get("exit_code"),
        }

    generated_files = payload.get("generated_files")
    compact_files = [
        {
            "path": str(item.get("path", "")),
            "relative_path": str(item.get("relative_path", "")),
            "written": bool(item.get("written", False)),
            "overwritten": bool(item.get("overwritten", False)),
            "bytes": int(item.get("bytes", 0)),
            **({"error": str(item["error"])} if item.get("error") else {}),
        }
        for item in generated_files
        if isinstance(item, dict)
    ] if isinstance(generated_files, list) else []

    executor_result = payload.get("executor_result")
    compact_executor_result = (
        {
            "validated": bool(executor_result.get("validated", False)),
            "llm_generated_code": bool(
                executor_result.get("llm_generated_code", False)
            ),
            "state_refs_consumed": list(
                executor_result.get("state_refs_consumed", [])
            ),
        }
        if isinstance(executor_result, dict)
        else {}
    )

    return {
        "stdout": _compact_text(str(payload.get("stdout", "")), text_limit),
        "stderr": _compact_text(
            str(payload.get("stderr", "")),
            max(120, text_limit // 2),
        ),
        "exit_code": payload.get("exit_code"),
        "latency_ms": payload.get("latency_ms"),
        "backend": payload.get("backend", ""),
        "executor_result": compact_executor_result,
        "generated_files": compact_files,
        "tool_feedback": payload.get("tool_feedback", {}),
        "codeact": codeact_summary,
    }
```

Full source code is retained because it is the executable non-text state needed
for downstream verification. Its bytes remain visible through
`state_transfer_bytes`; it must not be hidden from later benchmark accounting.

- [ ] **Step 4: Run all CodeAct and protocol-state tests**

Run:

```bash
uv run pytest tests/test_dynamic_agent_routing.py tests/test_real_agent_protocol_mode.py tests/test_sandbox_runner.py tests/test_agent_state_refs.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run AgentShell answer-fidelity tests**

Run:

```bash
uv run pytest tests/test_shell.py -q
```

Expected: all tests pass; code brackets and generated-file paths still render
without Rich markup loss.

- [ ] **Step 6: Commit the CodeAct schema fix**

```bash
git add src/agentmesh/modes/protocol_mode.py tests/test_dynamic_agent_routing.py tests/test_real_agent_protocol_mode.py
git commit -m "fix(state): preserve codeact result contract"
```

### Task 4: Separate Vector Capability from Semantic Claims

**Files:**
- Modify: `src/agentmesh/state/embedding.py:21-30`
- Modify: `src/agentmesh/memory/sqlite_store.py:124-235`
- Modify: `tests/test_memory_store.py`
- Modify: `tests/test_memory_rust_vector_index.py`

- [ ] **Step 1: Add capability tests**

Add a function that distinguishes vector availability from semantic quality:

```python
def supports_vector_ranking(encoder: object) -> bool:
    return isinstance(encoder, (HashEmbeddingEncoder, TEIEmbeddingEncoder))
```

Add tests:

```python
def test_hash_embedding_supports_lexical_vector_ranking() -> None:
    assert supports_vector_ranking(HashEmbeddingEncoder())
    assert not is_semantic_encoder(HashEmbeddingEncoder())
```

- [ ] **Step 2: Add Chinese lexical-vector ranking coverage**

Keep the existing `留学选校因素` regression and add an assertion that the result
reason identifies the source:

```python
results = memory_store.semantic_search_with_scores("留学选校因素", limit=1)

assert results[0].memory.memory_id == relevant.memory_id
assert "hash_vector" in results[0].reason
```

- [ ] **Step 3: Run the focused memory tests and observe both failures**

Run:

```bash
uv run pytest tests/test_memory_store.py::test_memory_store_semantic_search_handles_chinese_query tests/test_memory_rust_vector_index.py::test_semantic_search_uses_rust_memory_ranking_when_available -q
```

Expected: the Chinese result is wrong and the fake Rust ranker is not called.

- [ ] **Step 4: Let hash embeddings use vector ranking without semantic labeling**

Import `supports_vector_ranking` in `sqlite_store.py`. Build the query vector and
stored vectors before selecting the fallback:

```python
query_embedding = self.encoder.encode(query)
units: list[MemoryUnit] = []
vectors: list[list[float]] = []
fts_rank: dict[str, int] = {}
for rank_pos, unit in enumerate(candidate_units):
    vector = self._embedding_payload(unit)
    if vector is not None:
        units.append(unit)
        vectors.append(vector)
        fts_rank[unit.memory_id] = rank_pos

if not supports_vector_ranking(self.encoder) or not vectors:
    return self._rank_by_fts_fallback(
        candidates=candidate_units,
        query=query,
        query_tags=query_tags,
        limit=limit,
        scorer=scorer,
    )
```

Use the existing Rust `memory_rank_top_k` path for both encoder types. Pass a
`vector_source` value into result construction:

```python
vector_source = (
    "semantic_vector"
    if is_semantic_encoder(self.encoder)
    else "hash_vector"
)
```

Append `source={vector_source}` to each result reason. Keep the numeric
`semantic_similarity` field for schema compatibility during P0; P2 will rename
and version the metric.

- [ ] **Step 5: Ensure the Python path has the same ranking behavior**

Run once with Rust available and once with these functions monkeypatched:

```python
monkeypatch.setattr(
    "agentmesh.memory.sqlite_store._rust_memory_rank_available",
    lambda: False,
)
monkeypatch.setattr(
    "agentmesh.memory.sqlite_store.rust_available",
    lambda: False,
)
```

Assert the same top memory ID for both paths. Exact floating-point score parity
is not required; ordering parity is required.

- [ ] **Step 6: Run the complete memory and Rust parity suites**

Run:

```bash
uv run pytest tests/test_memory_store.py tests/test_memory_rust_vector_index.py tests/test_rust_embedding_parity.py tests/test_hybrid_memory.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the ranking fix**

```bash
git add src/agentmesh/state/embedding.py src/agentmesh/memory/sqlite_store.py tests/test_memory_store.py tests/test_memory_rust_vector_index.py
git commit -m "fix(memory): rank hash embeddings through vector path"
```

### Task 5: Verify the Complete P0 Runtime Contract

**Files:**
- Modify: `tests/test_modes_and_benchmark.py`
- Modify: `tests/test_shell.py`
- Test: all files under `tests/`

- [ ] **Step 1: Add a ten-round isolation regression**

Add a deterministic test that runs ten protocol tasks under one `RuntimePaths`
root:

```python
def test_protocol_mode_runs_ten_continuous_tasks_without_cross_trace_state(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(root=tmp_path)
    trace_ids: set[str] = set()

    for index in range(10):
        task = tmp_path / f"task-{index}.txt"
        task.write_text(
            f"Analyze protocol state transfer round {index}.",
            encoding="utf-8",
        )
        result = run_protocol_mode(
            task_path=task,
            paths=paths,
            load_configured_llm=False,
        )
        assert result.answer
        assert result.trace_id not in trace_ids
        trace_ids.add(result.trace_id)

    assert len(trace_ids) == 10
    message_trace_ids = {
        str(item["trace_id"])
        for item in read_jsonl(paths.protocol_messages)
    }
    assert trace_ids <= message_trace_ids
```

- [ ] **Step 2: Add error-continuation coverage to AgentShell**

Monkeypatch `run_protocol_with_progress` to raise once and succeed on the next
call. Verify:

```python
assert session.handle_line("first request")
assert session.handle_line("second request")
assert "command failed:" in output.getvalue()
assert "second answer" in output.getvalue()
```

This protects the interactive experience while runtime internals change.

- [ ] **Step 3: Run the P0-focused suite**

Run:

```bash
uv run pytest tests/test_capability_handshake.py tests/test_dynamic_agent_routing.py tests/test_memory_rust_vector_index.py tests/test_memory_store.py tests/test_real_agent_protocol_mode.py tests/test_modes_and_benchmark.py tests/test_shell.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run lint and type checking**

Run:

```bash
uv run ruff check .
uv run mypy src
```

Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 5: Run the complete test suite**

Run:

```bash
uv run pytest
```

Expected: all 160 existing tests plus the new P0 tests pass, with zero
failures.

- [ ] **Step 6: Run AgentShell smoke commands**

Run:

```bash
uv run agentmesh shell
```

Enter:

```text
/help
/config
/compare --no-llm Analyze structured protocol savings.
Write Python code that prints items[0].
/dashboard
/exit
```

Expected:

- Help and config render without exposing API keys.
- Compare shows both final answers, aligned Agent outputs, and metrics.
- Bare text routes through Protocol Mode and streams Agent output.
- Literal brackets in code remain intact.
- Dashboard path is printed.
- `/exit` terminates cleanly.

- [ ] **Step 7: Commit the P0 regression suite**

```bash
git add tests/test_modes_and_benchmark.py tests/test_shell.py
git commit -m "test(runtime): cover ten-round and shell recovery contracts"
```

### Task 6: Record the Verified Baseline

**Files:**
- Modify: `docs/progress.md`
- Test: `docs/progress.md`

- [ ] **Step 1: Update only verified facts**

Replace the stale test-count paragraph with the exact command results from Task
5. Add a dated P0 section containing:

```markdown
## 2026-06-30 P0 Engineering Baseline

- In-process AgentShell runs skip handshake overhead by default; formal
  evidence runs explicitly enable and record the complete handshake.
- Persisted CodeAct state preserves executable code, executor metadata, and
  generated-file write status.
- Hash embeddings use deterministic lexical-vector ranking and are not
  described as semantic embeddings.
- AgentShell bare-text routing, streaming, code fidelity, command surface, and
  error recovery are covered by regression tests.
- Ten continuous Protocol Mode tasks complete with distinct traces.
```

Insert the actual test count and command output summary after running Task 5;
do not copy the previous `45 tests passed` statement.

- [ ] **Step 2: Check documentation formatting and repository status**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only `docs/progress.md` is uncommitted.

- [ ] **Step 3: Commit the verified baseline record**

```bash
git add docs/progress.md
git commit -m "docs: record verified p0 engineering baseline"
```

## P0 Exit Gate

Do not begin P1 benchmark redesign until every condition is true:

- [ ] `uv run ruff check .` passes.
- [ ] `uv run mypy src` passes.
- [ ] `uv run pytest` has zero failures.
- [ ] Ten continuous Protocol Mode tasks pass.
- [ ] AgentShell command surface, bare-text routing, streaming, answer fidelity,
      dashboard generation, and error recovery pass.
- [ ] Protocol handshake messages are present in protocol evidence.
- [ ] CodeAct and real-LLM state payloads use the same persisted schema.
- [ ] Rust and Python memory-ranking paths select the same top result.
- [ ] Working tree is clean after the final P0 commit.
