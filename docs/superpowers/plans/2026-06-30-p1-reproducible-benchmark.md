# P1 Reproducible Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every benchmark result reproducible, track-isolated, statistically meaningful, and traceable to an immutable experiment manifest without changing the AgentShell interaction contract.

**Architecture:** Add experiment metadata and distribution statistics as focused evaluation modules. Store deterministic and LLM artifacts in separate track directories under each suite, while keeping compatibility readers for legacy artifacts. The benchmark runner remains the orchestrator but delegates manifest creation, paired order generation, and statistics to the new modules.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, SQLite, pytest, Rich, static HTML dashboard

---

## File Map

- Create `src/agentmesh/eval/experiment.py`: schema version, track enum,
  immutable experiment manifest, suite hash, task hash, and paired mode order.
- Create `src/agentmesh/eval/statistics.py`: mean, standard deviation, P50, and
  P95 calculation.
- Modify `src/agentmesh/storage/paths.py`: track-scoped benchmark artifact
  paths and manifest path.
- Modify `src/agentmesh/eval/benchmark.py`: structured detail records,
  manifest generation, track isolation, repeat metadata, and distributions.
- Modify `src/agentmesh/eval/report.py`: read the selected track and show
  experiment provenance/statistics.
- Modify `src/agentmesh/eval/dashboard.py`: discover track-scoped and legacy
  suites without mixing them.
- Modify `src/agentmesh/cli.py`, `src/agentmesh/shell/session.py`, and
  `src/agentmesh/shell/render.py`: pass and display track identity while
  preserving all existing AgentShell commands.
- Modify `tests/test_modes_and_benchmark.py`,
  `tests/test_benchmark_dashboard.py`, and `tests/test_shell.py`.

### Task 1: Add Experiment Identity and Safe Manifest

**Files:**
- Create: `src/agentmesh/eval/experiment.py`
- Create: `tests/test_experiment_manifest.py`

- [ ] **Step 1: Write manifest and pairing tests**

```python
from pathlib import Path

from agentmesh.eval.experiment import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkTrack,
    build_experiment_manifest,
    paired_mode_order,
)


def test_manifest_is_stable_and_does_not_contain_secrets(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: stable\nrepeat: 2\nseed: 17\ntasks: []\n",
        encoding="utf-8",
    )

    first = build_experiment_manifest(
        suite_path=suite,
        suite_name="stable",
        track=BenchmarkTrack.DETERMINISTIC,
        repeat_count=2,
        seed=17,
        environment={"model": "demo", "api_key": "secret"},
    )
    second = build_experiment_manifest(
        suite_path=suite,
        suite_name="stable",
        track=BenchmarkTrack.DETERMINISTIC,
        repeat_count=2,
        seed=17,
        environment={"model": "demo", "api_key": "different"},
    )

    assert first.schema_version == BENCHMARK_SCHEMA_VERSION
    assert first.experiment_id == second.experiment_id
    assert first.environment == {"model": "demo"}
    assert "secret" not in first.model_dump_json()


def test_paired_mode_order_is_deterministic_and_balanced() -> None:
    orders = [paired_mode_order(seed=17, pair_index=index) for index in range(10)]

    assert orders == [paired_mode_order(seed=17, pair_index=index) for index in range(10)]
    assert orders.count(("text", "protocol")) == 5
    assert orders.count(("protocol", "text")) == 5
```

- [ ] **Step 2: Run the tests and verify module import failure**

Run:

```bash
uv run pytest tests/test_experiment_manifest.py -q
```

Expected: collection fails because `agentmesh.eval.experiment` does not exist.

- [ ] **Step 3: Implement the experiment model**

```python
from __future__ import annotations

import hashlib
import platform
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

BENCHMARK_SCHEMA_VERSION = "2.0"
_SECRET_KEYS = {"api_key", "token", "secret", "password", "authorization"}


class BenchmarkTrack(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = BENCHMARK_SCHEMA_VERSION
    experiment_id: str
    suite_name: str
    suite_sha256: str
    track: BenchmarkTrack
    repeat_count: int
    seed: int
    python_version: str
    platform: str
    environment: dict[str, str]


def build_experiment_manifest(
    *,
    suite_path: Path,
    suite_name: str,
    track: BenchmarkTrack,
    repeat_count: int,
    seed: int,
    environment: dict[str, Any] | None = None,
) -> ExperimentManifest:
    suite_bytes = suite_path.read_bytes()
    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    safe_environment = {
        str(key): str(value)
        for key, value in sorted((environment or {}).items())
        if key.lower() not in _SECRET_KEYS and value is not None
    }
    identity = "|".join(
        [
            BENCHMARK_SCHEMA_VERSION,
            suite_name,
            suite_sha256,
            track.value,
            str(repeat_count),
            str(seed),
        ]
    )
    experiment_id = f"exp-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    return ExperimentManifest(
        experiment_id=experiment_id,
        suite_name=suite_name,
        suite_sha256=suite_sha256,
        track=track,
        repeat_count=repeat_count,
        seed=seed,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        environment=safe_environment,
    )


def paired_mode_order(*, seed: int, pair_index: int) -> tuple[str, str]:
    return (
        ("text", "protocol")
        if (seed + pair_index) % 2 == 0
        else ("protocol", "text")
    )
```

- [ ] **Step 4: Run manifest tests**

Run:

```bash
uv run pytest tests/test_experiment_manifest.py -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentmesh/eval/experiment.py tests/test_experiment_manifest.py
git commit -m "feat(benchmark): add reproducible experiment manifest"
```

### Task 2: Add Distribution Statistics

**Files:**
- Create: `src/agentmesh/eval/statistics.py`
- Create: `tests/test_benchmark_statistics.py`

- [ ] **Step 1: Write deterministic statistics tests**

```python
from agentmesh.eval.statistics import DistributionStats, distribution_stats


def test_distribution_stats_reports_mean_std_p50_and_p95() -> None:
    stats = distribution_stats([10, 20, 30, 40, 50])

    assert stats == DistributionStats(
        count=5,
        mean=30.0,
        stddev=14.142135623730951,
        p50=30.0,
        p95=48.0,
        minimum=10.0,
        maximum=50.0,
    )


def test_distribution_stats_handles_empty_and_single_samples() -> None:
    assert distribution_stats([]) == DistributionStats()
    assert distribution_stats([7]) == DistributionStats(
        count=1,
        mean=7.0,
        stddev=0.0,
        p50=7.0,
        p95=7.0,
        minimum=7.0,
        maximum=7.0,
    )
```

- [ ] **Step 2: Run tests and verify module import failure**

Run:

```bash
uv run pytest tests/test_benchmark_statistics.py -q
```

Expected: collection fails because `agentmesh.eval.statistics` does not exist.

- [ ] **Step 3: Implement statistics without a new dependency**

```python
import math
import statistics

from pydantic import BaseModel


class DistributionStats(BaseModel):
    count: int = 0
    mean: float = 0.0
    stddev: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0


def distribution_stats(values: list[int | float]) -> DistributionStats:
    samples = sorted(float(value) for value in values)
    if not samples:
        return DistributionStats()
    return DistributionStats(
        count=len(samples),
        mean=statistics.fmean(samples),
        stddev=statistics.pstdev(samples),
        p50=_percentile(samples, 0.50),
        p95=_percentile(samples, 0.95),
        minimum=samples[0],
        maximum=samples[-1],
    )


def _percentile(samples: list[float], quantile: float) -> float:
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    fraction = position - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction
```

- [ ] **Step 4: Run statistics tests**

Run:

```bash
uv run pytest tests/test_benchmark_statistics.py -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentmesh/eval/statistics.py tests/test_benchmark_statistics.py
git commit -m "feat(benchmark): add distribution statistics"
```

### Task 3: Isolate Artifacts by Track

**Files:**
- Modify: `src/agentmesh/storage/paths.py`
- Modify: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Write path isolation tests**

```python
def test_benchmark_artifacts_are_isolated_by_track(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)

    deterministic = paths.benchmark_suite_summary("suite", track="deterministic")
    llm = paths.benchmark_suite_summary("suite", track="llm")

    assert deterministic != llm
    assert deterministic.as_posix().endswith(
        "benchmarks/suite/deterministic/benchmark_summary.csv"
    )
    assert llm.as_posix().endswith("benchmarks/suite/llm/benchmark_summary.csv")
    assert paths.benchmark_suite_manifest("suite", track="llm").name == "manifest.json"
```

- [ ] **Step 2: Run the path test and verify signature failure**

Run:

```bash
uv run pytest tests/test_modes_and_benchmark.py::test_benchmark_artifacts_are_isolated_by_track -q
```

Expected: failure because path helpers do not accept `track`.

- [ ] **Step 3: Add optional track path parameters**

```python
def benchmark_suite_dir(self, suite_name: str, track: str | None = None) -> Path:
    suite_dir = self.benchmark_dir / _slugify_path_name(suite_name)
    return suite_dir / _slugify_path_name(track) if track else suite_dir

def benchmark_suite_summary(
    self,
    suite_name: str,
    track: str | None = None,
) -> Path:
    return self.benchmark_suite_dir(suite_name, track) / "benchmark_summary.csv"

def benchmark_suite_detail(
    self,
    suite_name: str,
    track: str | None = None,
) -> Path:
    return self.benchmark_suite_dir(suite_name, track) / "benchmark_detail.jsonl"

def benchmark_suite_report(
    self,
    suite_name: str,
    track: str | None = None,
) -> Path:
    return self.benchmark_suite_dir(suite_name, track) / "experiment_report.md"

def benchmark_suite_manifest(self, suite_name: str, track: str) -> Path:
    return self.benchmark_suite_dir(suite_name, track) / "manifest.json"
```

Calls without `track` retain the legacy path for compatibility readers.

- [ ] **Step 4: Run storage and path tests**

Run:

```bash
uv run pytest tests/test_modes_and_benchmark.py::test_benchmark_artifacts_are_isolated_by_track tests/test_state_store.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentmesh/storage/paths.py tests/test_modes_and_benchmark.py
git commit -m "feat(benchmark): isolate artifacts by execution track"
```

### Task 4: Emit Structured Detail and Statistical Summary

**Files:**
- Modify: `src/agentmesh/eval/benchmark.py`
- Modify: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Add benchmark provenance tests**

Create a two-task, two-repeat suite with `seed: 17`, run it offline, and assert:

```python
assert summary.schema_version == "2.0"
assert summary.track == "deterministic"
assert summary.repeat_count == 2
assert summary.total_runs == 4
assert summary.text_latency_stats.count == 4
assert summary.protocol_latency_stats.count == 4

manifest = orjson.loads(
    paths.benchmark_suite_manifest(
        "paired_suite",
        track="deterministic",
    ).read_bytes()
)
assert manifest["experiment_id"] == summary.experiment_id
assert manifest["seed"] == 17

detail = read_jsonl(
    paths.benchmark_suite_detail(
        "paired_suite",
        track="deterministic",
    )
)
assert len(detail) == 8
assert {row["repeat_index"] for row in detail} == {1, 2}
assert {tuple(row["pair_order"]) for row in detail} == {
    ("text", "protocol"),
    ("protocol", "text"),
}
assert all(row["task_sha256"] for row in detail)
```

- [ ] **Step 2: Run the test and verify missing fields/artifacts**

Run:

```bash
uv run pytest tests/test_modes_and_benchmark.py::test_benchmark_records_reproducible_provenance_and_statistics -q
```

Expected: failure because schema, manifest, track paths, and statistics are not
implemented.

- [ ] **Step 3: Extend summary and add detail model**

Add to `BenchmarkSummary`:

```python
schema_version: str = BENCHMARK_SCHEMA_VERSION
experiment_id: str
track: BenchmarkTrack
repeat_count: int
seed: int
text_latency_stats: DistributionStats
protocol_latency_stats: DistributionStats
text_token_stats: DistributionStats
protocol_token_stats: DistributionStats
```

Add:

```python
class BenchmarkDetailRecord(BaseModel):
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    experiment_id: str
    track: BenchmarkTrack
    repeat_index: int
    pair_index: int
    pair_order: tuple[str, str]
    task_id: str
    task_sha256: str
    group: str = ""
    topic: str = ""
    tags: list[str] = []
    depends_on: list[str] = []
    cold_start: bool = False
    mode: str
    trace_id: str
    metrics: RunMetrics
```

Use `Field(default_factory=list)` for list fields in production code.

- [ ] **Step 4: Build manifest and track paths at benchmark start**

Derive:

```python
track = BenchmarkTrack.LLM if use_llm else BenchmarkTrack.DETERMINISTIC
seed = int(suite.get("seed", 0))
runtime_config = AgentMeshConfig.from_project_root(paths.root)
manifest = build_experiment_manifest(
    suite_path=suite_path,
    suite_name=suite_name,
    track=track,
    repeat_count=repeat,
    seed=seed,
    environment={
        "model": runtime_config.llm.model or "",
        "rust_core": str(rust_available()),
    },
)
detail_path = paths.benchmark_suite_detail(suite_name, track=track.value)
summary_path = paths.benchmark_suite_summary(suite_name, track=track.value)
report_path = paths.benchmark_suite_report(suite_name, track=track.value)
manifest_path = paths.benchmark_suite_manifest(suite_name, track=track.value)
```

Write manifest with `orjson.dumps(manifest.model_dump(mode="json"),
option=orjson.OPT_INDENT_2)`.

- [ ] **Step 5: Execute each pair in deterministic balanced order**

For each logical pair:

```python
pair_index = _round * len(tasks) + task_index - 1
order = paired_mode_order(seed=seed, pair_index=pair_index)
results: dict[str, ModeRunResult] = {}
for mode in order:
    results[mode] = (
        run_text_mode(
            task_path=input_file,
            paths=paths,
            load_configured_llm=use_llm,
        )
        if mode == "text"
        else run_protocol_mode(
            task_path=input_file,
            paths=paths,
            load_configured_llm=use_llm,
        )
    )
text_result = results["text"]
protocol_result = results["protocol"]
```

Progress stages must use actual execution order, while aggregation remains
paired by task.

- [ ] **Step 6: Record per-sample values and structured detail**

Maintain four lists:

```python
text_latency_samples: list[int] = []
protocol_latency_samples: list[int] = []
text_token_samples: list[int] = []
protocol_token_samples: list[int] = []
```

Append each pair and serialize `BenchmarkDetailRecord` instead of an anonymous
dictionary. Compute task hash from exact input bytes.

- [ ] **Step 7: Populate distribution fields and run tests**

Run:

```bash
uv run pytest tests/test_modes_and_benchmark.py -q
```

Expected: all benchmark tests pass with track-scoped paths and provenance.

- [ ] **Step 8: Commit**

```bash
git add src/agentmesh/eval/benchmark.py tests/test_modes_and_benchmark.py
git commit -m "feat(benchmark): record paired provenance and statistics"
```

### Task 5: Update Report, Dashboard, CLI, and AgentShell

**Files:**
- Modify: `src/agentmesh/eval/report.py`
- Modify: `src/agentmesh/eval/dashboard.py`
- Modify: `src/agentmesh/cli.py`
- Modify: `src/agentmesh/shell/session.py`
- Modify: `src/agentmesh/shell/render.py`
- Modify: `tests/test_benchmark_dashboard.py`
- Modify: `tests/test_shell.py`

- [ ] **Step 1: Write track-aware report and dashboard tests**

Assert that deterministic and LLM runs for one suite both remain discoverable:

```python
data = load_dashboard_data(tmp_path)
identities = {
    (suite["name"], suite["track"])
    for suite in data["suites"]
    if suite["status"] == "generated"
}
assert ("same_suite", "deterministic") in identities
assert ("same_suite", "llm") in identities
```

Assert the report contains:

```python
assert "SchemaVersion: 2.0" in report
assert "Track: deterministic" in report
assert "ExperimentId: exp-" in report
assert "Latency P50/P95" in report
```

- [ ] **Step 2: Run UI/report tests and verify failures**

Run:

```bash
uv run pytest tests/test_benchmark_dashboard.py tests/test_shell.py tests/test_modes_and_benchmark.py -q
```

Expected: failures identify legacy-only path assumptions.

- [ ] **Step 3: Make report generation track-aware**

Change signature:

```python
def generate_report(
    paths: RuntimePaths,
    *,
    suite_name: str | None = None,
    track: str | None = None,
) -> Path:
```

Use track-scoped summary/detail/report paths when both values are supplied.
Keep the legacy branch when `track is None`.

- [ ] **Step 4: Discover track directories in dashboard loader**

For each configured suite directory:

1. Load direct legacy artifacts if present and mark track `"legacy"`.
2. Inspect `deterministic/` and `llm/` children.
3. Create one dashboard suite entry per generated track.
4. Never aggregate rows across tracks.
5. Show schema mismatch as a diagnostic instead of treating missing values as
   zero.

- [ ] **Step 5: Pass track from CLI and AgentShell**

Update both benchmark callers:

```python
report_path = generate_report(
    paths,
    suite_name=summary.suite_name,
    track=summary.track.value,
)
```

Render `track`, `experiment_id`, repeat count, and latency P50/P95 in the
benchmark table without changing existing commands or bare-text routing.

- [ ] **Step 6: Run AgentShell and dashboard regression**

Run:

```bash
uv run pytest tests/test_benchmark_dashboard.py tests/test_shell.py tests/test_modes_and_benchmark.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/agentmesh/eval/report.py src/agentmesh/eval/dashboard.py src/agentmesh/cli.py src/agentmesh/shell/session.py src/agentmesh/shell/render.py tests/test_benchmark_dashboard.py tests/test_shell.py
git commit -m "feat(benchmark): expose track provenance in reports and shell"
```

### Task 6: P1 Verification and Documentation

**Files:**
- Modify: `docs/benchmark_design.md`
- Modify: `docs/progress.md`

- [ ] **Step 1: Document the new artifact layout**

```text
runs/latest/benchmarks/<suite>/<track>/
  manifest.json
  benchmark_summary.csv
  benchmark_detail.jsonl
  experiment_report.md
```

Document schema version `2.0`, deterministic versus LLM tracks, seed, balanced
paired ordering, task hashes, and distribution statistics.

- [ ] **Step 2: Run one small offline benchmark**

Run:

```bash
uv run agentmesh benchmark --suite standard --no-llm
```

Expected: deterministic track artifacts and manifest are generated; LLM track
artifacts are not overwritten.

- [ ] **Step 3: Run full quality gates**

Run:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 4: Run AgentShell regression suite explicitly**

Run:

```bash
uv run pytest tests/test_shell.py tests/test_compare_prompt.py tests/test_interactive_agent.py -q
```

Expected: all AgentShell interaction tests pass.

- [ ] **Step 5: Record exact verified results and commit**

Update `docs/progress.md` using actual command output, then:

```bash
git add docs/benchmark_design.md docs/progress.md
git commit -m "docs: record reproducible benchmark p1"
```

## P1 Exit Gate

- [ ] Deterministic and LLM tracks never overwrite one another.
- [ ] Every summary and detail row has schema and experiment identity.
- [ ] Manifest contains no API key or secret value.
- [ ] Pair execution order is deterministic and balanced.
- [ ] Summary includes mean, standard deviation, P50, and P95 sample statistics.
- [ ] Dashboard and report never combine different tracks.
- [ ] Legacy artifacts remain readable and are clearly labeled.
- [ ] AgentShell command surface, streaming, answer fidelity, and error recovery
      remain unchanged.
- [ ] Ruff, mypy, and the complete pytest suite pass.
