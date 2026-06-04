# Rust Runtime Core Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor AgentMesh Runtime into a Python agent layer plus Rust runtime core so Protocol Mode keeps its token and memory advantages while improving latency, state throughput, and semantic-search scalability.

**Architecture:** Keep Python for CLI, deterministic agent strategies, prompts, reports, and LLM integration. Add a PyO3-based Rust extension named `agentmesh_core` for hot-path runtime primitives: AMP codec, StateRef parsing, hash embedding, cosine top-k, state journal operations, and sandbox process management. Migrate incrementally behind Python fallbacks so every task is testable and reversible.

**Tech Stack:** Python 3.11, PyO3, maturin, Rust 2021, serde, serde_json, rmp-serde, sha2, rusqlite, tempfile, existing pytest/ruff/mypy benchmark flow.

---

## File Structure

- Modify: `pyproject.toml`
  - Add `maturin` to dev dependencies.
  - Keep the existing Python package entrypoint unchanged.
- Create: `Cargo.toml`
  - Workspace root for Rust crates.
- Create: `crates/agentmesh-core/Cargo.toml`
  - PyO3 extension crate metadata and dependencies.
- Create: `crates/agentmesh-core/src/lib.rs`
  - Python module exports.
- Create: `crates/agentmesh-core/src/state_ref.rs`
  - Rust StateRef parser and validator.
- Create: `crates/agentmesh-core/src/embedding.rs`
  - Rust hash embedding and cosine similarity.
- Create: `crates/agentmesh-core/src/codec.rs`
  - Rust AMP JSON and msgpack codec helpers.
- Create: `crates/agentmesh-core/src/vector_index.rs`
  - Rust top-k semantic search over dense `f32` vectors.
- Create: `crates/agentmesh-core/src/sandbox_pool.rs`
  - Optional later-stage warmed Python worker pool.
- Create: `src/agentmesh/core.py`
  - Python import wrapper for optional Rust extension and feature flags.
- Modify: `src/agentmesh/state/refs.py`
  - Use Rust StateRef parser when available.
- Modify: `src/agentmesh/state/embedding.py`
  - Use Rust embedding and cosine functions when available.
- Modify: `src/agentmesh/protocol/codec.py`
  - Use Rust codec when available.
- Modify: `src/agentmesh/memory/sqlite_store.py`
  - Use Rust vector top-k for semantic search when available.
- Modify: `src/agentmesh/state/store.py`
  - Stop replaying full `protocol/states.jsonl` on every initialization; use SQLite indexed records and current-trace cache.
- Modify: `src/agentmesh/modes/protocol_mode.py`
  - Add stage timing metrics and use optimized stores.
- Modify: `src/agentmesh/eval/metrics.py`
  - Add optional stage latency fields for protocol profiling.
- Create: `tests/test_core_optional.py`
  - Verify Python fallback works when Rust extension is absent.
- Create: `tests/test_rust_embedding_parity.py`
  - Verify Rust embedding matches Python behavior.
- Create: `tests/test_rust_codec_parity.py`
  - Verify Rust codec matches Python AMP roundtrip behavior.
- Create: `tests/test_state_store_fast_index.py`
  - Verify StateStore no longer depends on full JSONL replay.
- Create: `tests/test_memory_rust_vector_index.py`
  - Verify Rust top-k search returns expected memory order.
- Create: `scripts/profile_protocol.py`
  - Repeatable latency profiling command for Protocol Mode stages.
- Modify: `docs/architecture.md`
  - Document Python layer + Rust core architecture.
- Modify: `docs/benchmark_design.md`
  - Document Rust optimized latency metrics and fairness rules.

---

## Task 1: Add Profiling Guardrails Before Refactor

**Files:**
- Modify: `src/agentmesh/eval/metrics.py`
- Modify: `src/agentmesh/modes/protocol_mode.py`
- Create: `scripts/profile_protocol.py`
- Test: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Extend metrics with stage timings**

In `src/agentmesh/eval/metrics.py`, add this field to `RunMetrics`:

```python
stage_latency_ms: dict[str, int] = {}
```

- [ ] **Step 2: Add a tiny timer helper in Protocol Mode**

In `src/agentmesh/modes/protocol_mode.py`, measure at least these stages:

```python
stage_latency_ms: dict[str, int] = {}

def mark_stage(name: str, stage_start: float) -> None:
    stage_latency_ms[name] = int((time.perf_counter() - stage_start) * 1000)
```

Record stages named:

```python
"setup"
"state_task_embedding"
"planner"
"memory_search"
"retriever"
"sandbox"
"executor"
"summarizer"
"memory_write"
"artifact_write"
```

Pass `stage_latency_ms=stage_latency_ms` into `RunMetrics`.

- [ ] **Step 3: Add a profiling script**

Create `scripts/profile_protocol.py`:

```python
from pathlib import Path

from agentmesh.modes.protocol_mode import run_protocol_mode
from agentmesh.storage.paths import RuntimePaths


def main() -> None:
    paths = RuntimePaths(root=Path("."))
    task = Path("examples/tasks/A1_requirements.txt")
    result = run_protocol_mode(task, paths)
    print(result.metrics.model_dump())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run profiling once**

Run:

```bash
uv run python scripts/profile_protocol.py
```

Expected: output includes `stage_latency_ms` with the stage names above.

- [ ] **Step 5: Run existing verification**

Run:

```bash
uv run pytest tests/test_modes_and_benchmark.py
```

Expected: all tests pass.

---

## Task 2: Fix StateStore Full-Replay Latency Before Rust Migration

**Files:**
- Modify: `src/agentmesh/state/store.py`
- Test: `tests/test_state_store_fast_index.py`

- [ ] **Step 1: Write failing test for fast startup**

Create `tests/test_state_store_fast_index.py`:

```python
from pathlib import Path

from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_state_store_can_reload_record_from_sqlite_index(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    first = StateStore(paths)
    ref = first.put_text("trace-1", "tester", "hello", metadata={"k": "v"})

    second = StateStore(paths)
    record, payload = second.get(ref)

    assert payload == "hello"
    assert record.trace_id == "trace-1"
    assert record.metadata == {"k": "v"}
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```bash
uv run pytest tests/test_state_store_fast_index.py -q
```

Expected: it may pass today, but observe that implementation still replays JSONL. Continue because the purpose is to preserve behavior while changing internals.

- [ ] **Step 3: Add `record_json` column to SQLite index**

In `src/agentmesh/state/store.py`, update `_init_index()` to add:

```sql
record_json TEXT NOT NULL DEFAULT '{}'
```

Use a migration-safe `ALTER TABLE` guarded by checking `PRAGMA table_info(state_records)`.

- [ ] **Step 4: Store full record JSON in `_upsert_index`**

Update `_upsert_index()` so each row stores:

```python
record.model_dump_json()
```

in `record_json`.

- [ ] **Step 5: Replace full JSONL replay with SQLite load**

Change `_load_records()` to read from `state_records.record_json` instead of `protocol/states.jsonl`.

Use:

```python
rows = conn.execute("SELECT record_json FROM state_records").fetchall()
```

Then:

```python
record = StateRecord.model_validate_json(str(row[0]))
self._records[record.state_id] = record
```

- [ ] **Step 6: Verify**

Run:

```bash
uv run pytest tests/test_state_store.py tests/test_state_store_fast_index.py -q
uv run python scripts/profile_protocol.py
```

Expected: tests pass and Protocol Mode setup/state stages no longer grow because of JSONL replay.

---

## Task 3: Add Rust Core Scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `Cargo.toml`
- Create: `crates/agentmesh-core/Cargo.toml`
- Create: `crates/agentmesh-core/src/lib.rs`
- Create: `src/agentmesh/core.py`
- Test: `tests/test_core_optional.py`

- [ ] **Step 1: Add optional Rust import wrapper**

Create `src/agentmesh/core.py`:

```python
from typing import Any


try:
    import agentmesh_core as _rust_core
except Exception:
    _rust_core = None


def rust_available() -> bool:
    return _rust_core is not None


def rust_core() -> Any:
    if _rust_core is None:
        raise RuntimeError("agentmesh_core Rust extension is not installed")
    return _rust_core
```

- [ ] **Step 2: Add fallback test**

Create `tests/test_core_optional.py`:

```python
from agentmesh.core import rust_available


def test_rust_core_is_optional() -> None:
    assert isinstance(rust_available(), bool)
```

- [ ] **Step 3: Add Rust workspace**

Create root `Cargo.toml`:

```toml
[workspace]
members = ["crates/agentmesh-core"]
resolver = "2"
```

- [ ] **Step 4: Add PyO3 crate manifest**

Create `crates/agentmesh-core/Cargo.toml`:

```toml
[package]
name = "agentmesh-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "agentmesh_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rmp-serde = "1"
sha2 = "0.10"
```

- [ ] **Step 5: Add minimal Rust module**

Create `crates/agentmesh-core/src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn agentmesh_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
```

- [ ] **Step 6: Add maturin dev dependency**

In `pyproject.toml`, add to `[project.optional-dependencies].dev`:

```toml
"maturin>=1.7.0",
```

- [ ] **Step 7: Build and test**

Run:

```bash
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
uv run pytest tests/test_core_optional.py -q
```

Expected: Rust extension builds, test passes.

---

## Task 4: Port HashEmbedding and Cosine Similarity to Rust

**Files:**
- Create: `crates/agentmesh-core/src/embedding.rs`
- Modify: `crates/agentmesh-core/src/lib.rs`
- Modify: `src/agentmesh/state/embedding.py`
- Test: `tests/test_rust_embedding_parity.py`

- [ ] **Step 1: Write parity test**

Create `tests/test_rust_embedding_parity.py`:

```python
from agentmesh.state.embedding import HashEmbeddingEncoder, cosine_similarity


def test_hash_embedding_is_deterministic() -> None:
    encoder = HashEmbeddingEncoder(dimensions=384)
    left = encoder.encode("agent state passing memory")
    right = encoder.encode("agent state passing memory")

    assert left == right
    assert len(left) == 384
    assert abs(cosine_similarity(left, right) - 1.0) < 1e-9
```

- [ ] **Step 2: Add Rust implementation**

Create `crates/agentmesh-core/src/embedding.rs` with exported functions:

```rust
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

#[pyfunction]
pub fn hash_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let tokens: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if tokens.is_empty() {
        return vec![0.0; dimensions];
    }
    let mut features = tokens.clone();
    for pair in tokens.windows(2) {
        features.push(format!("{}_{}", pair[0], pair[1]));
    }
    let mut vector = vec![0.0f32; dimensions];
    for token in features {
        let digest = Sha256::digest(token.as_bytes());
        let bucket = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % dimensions;
        let sign = if digest[4] % 2 == 0 { 1.0 } else { -1.0 };
        vector[bucket] += sign;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[pyfunction]
pub fn cosine_similarity_f32(left: Vec<f32>, right: Vec<f32>) -> f32 {
    if left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}
```

- [ ] **Step 3: Export Rust functions**

In `crates/agentmesh-core/src/lib.rs`:

```rust
mod embedding;

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn agentmesh_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::hash_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::cosine_similarity_f32, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Use Rust from Python with fallback**

In `src/agentmesh/state/embedding.py`, keep the existing Python implementation and branch inside `encode()`:

```python
from agentmesh.core import rust_available, rust_core

if rust_available():
    return [float(value) for value in rust_core().hash_embedding(text, self.dimensions)]
```

Update `cosine_similarity()` similarly:

```python
if rust_available():
    return float(rust_core().cosine_similarity_f32(left, right))
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
uv run pytest tests/test_embedding_ref.py tests/test_rust_embedding_parity.py -q
```

Expected: tests pass.

---

## Task 5: Add Rust Vector Top-K for Memory Semantic Search

**Files:**
- Create: `crates/agentmesh-core/src/vector_index.rs`
- Modify: `crates/agentmesh-core/src/lib.rs`
- Modify: `src/agentmesh/memory/sqlite_store.py`
- Test: `tests/test_memory_rust_vector_index.py`

- [ ] **Step 1: Add test for semantic order**

Create `tests/test_memory_rust_vector_index.py`:

```python
from pathlib import Path

from agentmesh.memory.schema import MemoryUnit
from agentmesh.memory.sqlite_store import SQLiteMemoryStore
from agentmesh.state.embedding import HashEmbeddingEncoder
from agentmesh.state.store import StateStore
from agentmesh.storage.paths import RuntimePaths


def test_semantic_search_prefers_related_memory(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    state_store = StateStore(paths)
    encoder = HashEmbeddingEncoder()
    store = SQLiteMemoryStore(paths, state_store, encoder)

    rust_ref = state_store.put_embedding("trace-1", "tester", encoder.encode("rust runtime state"))
    docs_ref = state_store.put_embedding("trace-1", "tester", encoder.encode("readme documentation"))

    store.put(MemoryUnit(
        source_agent="tester",
        task_topic="rust runtime",
        summary="Rust runtime state passing optimization",
        tags=["runtime"],
        embedding_ref=rust_ref,
        provenance_trace_id="trace-1",
    ))
    store.put(MemoryUnit(
        source_agent="tester",
        task_topic="documentation",
        summary="README documentation writing",
        tags=["docs"],
        embedding_ref=docs_ref,
        provenance_trace_id="trace-1",
    ))

    results = store.semantic_search("state runtime optimization", limit=1)

    assert results[0].task_topic == "rust runtime"
```

- [ ] **Step 2: Add Rust top-k implementation**

Create `crates/agentmesh-core/src/vector_index.rs`:

```rust
use pyo3::prelude::*;

#[pyfunction]
pub fn top_k_cosine(query: Vec<f32>, vectors: Vec<Vec<f32>>, limit: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| (cosine(&query, vector), idx))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(limit).map(|(_, idx)| idx).collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}
```

- [ ] **Step 3: Export top-k**

In `crates/agentmesh-core/src/lib.rs`, add:

```rust
mod vector_index;
```

and:

```rust
m.add_function(wrap_pyfunction!(vector_index::top_k_cosine, m)?)?;
```

- [ ] **Step 4: Use Rust top-k in `semantic_search`**

In `src/agentmesh/memory/sqlite_store.py`, collect units and payload vectors as today. If Rust is available, call:

```python
indexes = rust_core().top_k_cosine(query_embedding, vectors, limit)
return [units[index] for index in indexes]
```

Keep the existing Python sort as fallback.

- [ ] **Step 5: Verify**

Run:

```bash
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
uv run pytest tests/test_memory_store.py tests/test_memory_rust_vector_index.py -q
```

Expected: tests pass.

---

## Task 6: Port StateRef Parser and AMP Codec to Rust

**Files:**
- Create: `crates/agentmesh-core/src/state_ref.rs`
- Create: `crates/agentmesh-core/src/codec.rs`
- Modify: `crates/agentmesh-core/src/lib.rs`
- Modify: `src/agentmesh/state/refs.py`
- Modify: `src/agentmesh/protocol/codec.py`
- Test: `tests/test_rust_codec_parity.py`

- [ ] **Step 1: Add codec parity test**

Create `tests/test_rust_codec_parity.py`:

```python
from agentmesh.protocol.codec import decode_message, encode_message
from agentmesh.protocol.enums import MsgType
from agentmesh.protocol.schema import AMPMessage
from agentmesh.state.refs import parse_state_ref


def test_codec_roundtrip_still_returns_amp_message() -> None:
    message = AMPMessage(
        trace_id="trace-1",
        source_agent="a",
        target_agent="b",
        msg_type=MsgType.INVOKE,
        action="plan.create",
        state_refs=["state://text/state-abc"],
    )

    decoded = decode_message(encode_message(message))

    assert decoded.trace_id == "trace-1"
    assert decoded.action == "plan.create"


def test_parse_state_ref_accepts_valid_ref() -> None:
    parsed = parse_state_ref("state://text/state-abc")

    assert parsed.state_type == "text"
    assert parsed.state_id == "state-abc"
```

- [ ] **Step 2: Add Rust StateRef parser**

Create `crates/agentmesh-core/src/state_ref.rs`:

```rust
use pyo3::prelude::*;

#[pyfunction]
pub fn parse_state_ref_parts(reference: &str) -> PyResult<(String, String)> {
    let rest = reference
        .strip_prefix("state://")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("StateRef must start with state://"))?;
    let mut parts = rest.splitn(2, '/');
    let state_type = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("StateRef missing type"))?;
    let state_id = parts
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("StateRef missing id"))?;
    Ok((state_type.to_string(), state_id.to_string()))
}
```

- [ ] **Step 3: Add Rust JSON codec helper**

Create `crates/agentmesh-core/src/codec.rs`:

```rust
use pyo3::prelude::*;

#[pyfunction]
pub fn encode_json_bytes(json_text: &str) -> PyResult<Vec<u8>> {
    let value: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    serde_json::to_vec(&value)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
}

#[pyfunction]
pub fn decode_json_text(data: &[u8]) -> PyResult<String> {
    let value: serde_json::Value = serde_json::from_slice(data)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    serde_json::to_string(&value)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
}
```

- [ ] **Step 4: Export functions and wire Python fallbacks**

Export the Rust functions in `lib.rs`, then update Python modules to call Rust when available and preserve current Python behavior when unavailable.

- [ ] **Step 5: Verify**

Run:

```bash
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
uv run pytest tests/test_protocol_schema.py tests/test_rust_codec_parity.py -q
```

Expected: tests pass.

---

## Task 7: Reduce Sandbox Cold-Start Cost

**Files:**
- Modify: `src/agentmesh/sandbox/runner.py`
- Create: `src/agentmesh/sandbox/worker.py`
- Test: `tests/test_sandbox_runner.py`

- [ ] **Step 1: Keep existing subprocess runner as fallback**

Do not remove `SandboxRunner.run_python`. Add a new optional `warm` mode later, defaulting to current behavior.

- [ ] **Step 2: Add worker script**

Create `src/agentmesh/sandbox/worker.py`:

```python
import contextlib
import io
import json
import sys


for line in sys.stdin:
    request = json.loads(line)
    code = request["code"]
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, {"__name__": "__sandbox__"})
    except Exception as exc:
        exit_code = 1
        print(repr(exc), file=stderr)
    print(json.dumps({
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "exit_code": exit_code,
    }), flush=True)
```

- [ ] **Step 3: Add explicit benchmark note**

Keep Protocol Mode using cold subprocess until warm worker is fully tested. This prevents changing CodeAct semantics before Rust core metrics are known.

- [ ] **Step 4: Verify current sandbox tests**

Run:

```bash
uv run pytest tests/test_sandbox_runner.py -q
```

Expected: existing tests pass.

---

## Task 8: Run Benchmark After Each Optimization Layer

**Files:**
- Modify: `docs/benchmark_design.md`
- No new test file.

- [ ] **Step 1: Run full quality checks**

Run:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Expected: ruff passes, mypy passes, pytest passes.

- [ ] **Step 2: Run benchmark**

Run:

```bash
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
uv run agentmesh report --run runs/latest
```

Expected:

```text
runs/latest/benchmark_summary.csv exists
runs/latest/experiment_report.md exists
```

- [ ] **Step 3: Compare target metrics**

Use current baseline:

```text
TokenSavingRate: about 0.7055
MemoryHitRate: about 0.9667
Protocol avg latency before refactor: about 9636 ms
```

Target after Tasks 2-6:

```text
TokenSavingRate >= 0.65
MemoryHitRate >= 0.90
Protocol avg latency materially lower than 9636 ms
LatencyReductionRate less negative, ideally near zero or positive
```

---

## Task 9: Documentation Update for Competition Narrative

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/benchmark_design.md`
- Modify: `README.md`

- [ ] **Step 1: Document architecture**

Add this architecture summary to `docs/architecture.md`:

```markdown
## Rust Core Optimization

The runtime keeps Python for agent policies, CLI, prompts, LLM integration, and reports. Hot-path system primitives are exposed through an optional `agentmesh_core` Rust extension:

- AMP codec and StateRef parsing
- HashEmbedding and cosine top-k semantic search
- Fast state index access
- Future sandbox worker management

This preserves reproducible Python-level behavior while reducing runtime overhead in Protocol Mode.
```

- [ ] **Step 2: Document benchmark fairness**

Add this to `docs/benchmark_design.md`:

```markdown
## Fairness Rules

Text Mode and Protocol Mode must run the same task suite, repeat count, deterministic agents, and sandbox behavior. Rust optimizations may reduce serialization, indexing, state-transfer, and retrieval overhead, but must not change the task content, memory write policy, or quality scoring function.
```

- [ ] **Step 3: Verify docs and tests**

Run:

```bash
uv run pytest
```

Expected: all tests pass.

---

## Task 10: Execution Order and Commit Strategy

Execute in this order:

1. Task 1: profiling guardrails.
2. Task 2: StateStore fast index.
3. Task 3: Rust scaffold.
4. Task 4: Rust embedding.
5. Task 5: Rust vector top-k.
6. Task 6: Rust StateRef and codec.
7. Task 8: full benchmark comparison.
8. Task 9: docs.
9. Task 7: sandbox warm worker only after metrics are stable.

Suggested commits:

```bash
git add src/agentmesh/eval/metrics.py src/agentmesh/modes/protocol_mode.py scripts/profile_protocol.py tests/test_modes_and_benchmark.py
git commit -m "chore: add protocol stage profiling"

git add src/agentmesh/state/store.py tests/test_state_store_fast_index.py
git commit -m "perf: load state records from sqlite index"

git add pyproject.toml Cargo.toml crates/agentmesh-core src/agentmesh/core.py tests/test_core_optional.py
git commit -m "feat: add optional rust runtime core"

git add crates/agentmesh-core/src/embedding.rs crates/agentmesh-core/src/lib.rs src/agentmesh/state/embedding.py tests/test_rust_embedding_parity.py
git commit -m "perf: move hash embedding to rust core"

git add crates/agentmesh-core/src/vector_index.rs crates/agentmesh-core/src/lib.rs src/agentmesh/memory/sqlite_store.py tests/test_memory_rust_vector_index.py
git commit -m "perf: add rust vector top-k memory search"

git add crates/agentmesh-core/src/state_ref.rs crates/agentmesh-core/src/codec.rs crates/agentmesh-core/src/lib.rs src/agentmesh/state/refs.py src/agentmesh/protocol/codec.py tests/test_rust_codec_parity.py
git commit -m "perf: add rust protocol parsing helpers"

git add docs/architecture.md docs/benchmark_design.md README.md
git commit -m "docs: describe rust core runtime optimization"
```

---

## Self-Review

- Spec coverage: The plan preserves the required Text Mode versus Protocol Mode comparison, keeps StateRef and shared memory behavior, and targets latency, state bytes, protocol bytes, and semantic search scalability.
- Placeholder scan: No task uses open-ended placeholder wording; every task names concrete files, commands, and expected outcomes.
- Type consistency: Python wrapper exposes `rust_available()` and `rust_core()`; later tasks use the same names. Rust exports use `hash_embedding`, `cosine_similarity_f32`, `top_k_cosine`, `parse_state_ref_parts`, `encode_json_bytes`, and `decode_json_text`.
