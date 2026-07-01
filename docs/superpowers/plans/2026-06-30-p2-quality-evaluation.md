# P2 Evidence-Based Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace answer-length scoring with task-defined, deterministic quality checks and expose quality averages, pass rates, and deltas without changing AgentShell behavior.

**Architecture:** Add a pure quality evaluator that consumes an answer and a versioned suite quality specification. Modes continue producing answers and operational metrics; the benchmark runner applies the same evaluator to paired Text/Protocol results before aggregation. Suites without an explicit quality rule are labeled `unscored` rather than receiving a fabricated score.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, regex, pytest

---

## File Map

- Create `src/agentmesh/eval/quality_spec.py`: quality rule schema and
  deterministic evaluator.
- Modify `src/agentmesh/eval/metrics.py`: nullable quality score, quality
  status, and rule identity.
- Modify `src/agentmesh/eval/benchmark.py`: paired evaluation and quality
  aggregate fields.
- Modify `src/agentmesh/eval/compare.py`: direct prompt comparisons report
  quality as unavailable unless an explicit evaluator is supplied.
- Modify `src/agentmesh/eval/report.py`,
  `src/agentmesh/eval/dashboard.py`, and `src/agentmesh/shell/render.py`:
  explain quality evidence.
- Modify `examples/benchmarks/continuous_tasks.yaml`,
  `examples/benchmarks/long_context_tasks.yaml`, and
  `examples/benchmarks/showcase_benchmark.yaml`: explicit quality rules.
- Modify quality, benchmark, dashboard, report, and AgentShell tests.

### Task 1: Implement Versioned Quality Rules

**Files:**
- Create: `src/agentmesh/eval/quality_spec.py`
- Create: `tests/test_quality_spec.py`

- [ ] **Step 1: Write evaluator tests**

```python
from agentmesh.eval.quality_spec import QualitySpec, evaluate_quality


def test_contains_all_quality_is_case_insensitive() -> None:
    spec = QualitySpec(
        rule_id="sum-output-v1",
        kind="contains_all",
        expected=["sum: 77", "even count: 5"],
    )

    result = evaluate_quality("Sum: 77\nEven count: 5", spec)

    assert result.scored
    assert result.passed
    assert result.score == 1.0
    assert result.matched == 2
    assert result.total == 2


def test_contains_all_quality_reports_partial_credit() -> None:
    spec = QualitySpec(
        rule_id="two-facts-v1",
        kind="contains_all",
        expected=["alpha", "beta"],
    )

    result = evaluate_quality("Alpha only", spec)

    assert result.scored
    assert not result.passed
    assert result.score == 0.5


def test_regex_and_non_empty_rules() -> None:
    regex = evaluate_quality(
        "result=42",
        QualitySpec(rule_id="number-v1", kind="regex", pattern=r"result\s*=\s*42"),
    )
    non_empty = evaluate_quality(
        "  ",
        QualitySpec(rule_id="answer-v1", kind="non_empty"),
    )

    assert regex.passed and regex.score == 1.0
    assert not non_empty.passed and non_empty.score == 0.0


def test_missing_quality_rule_is_unscored() -> None:
    result = evaluate_quality("arbitrary answer", None)

    assert not result.scored
    assert result.score is None
    assert result.reason == "quality rule not configured"
```

- [ ] **Step 2: Run tests and verify module import failure**

Run:

```bash
uv run pytest tests/test_quality_spec.py -q
```

Expected: collection fails because `quality_spec` does not exist.

- [ ] **Step 3: Implement the evaluator**

```python
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QualitySpec(BaseModel):
    version: str = "1.0"
    rule_id: str
    kind: Literal["non_empty", "contains_all", "regex"]
    expected: list[str] = Field(default_factory=list)
    pattern: str = ""
    pass_threshold: float = 1.0

    @model_validator(mode="after")
    def validate_rule(self) -> "QualitySpec":
        if self.kind == "contains_all" and not self.expected:
            raise ValueError("contains_all requires expected values")
        if self.kind == "regex" and not self.pattern:
            raise ValueError("regex requires pattern")
        return self


class QualityResult(BaseModel):
    rule_id: str = "unscored"
    scored: bool = False
    passed: bool = False
    score: float | None = None
    matched: int = 0
    total: int = 0
    reason: str = ""


def evaluate_quality(answer: str, spec: QualitySpec | None) -> QualityResult:
    if spec is None:
        return QualityResult(reason="quality rule not configured")
    if spec.kind == "non_empty":
        score = 1.0 if answer.strip() else 0.0
        return _result(spec, score, int(score), 1)
    if spec.kind == "regex":
        matched = int(re.search(spec.pattern, answer, flags=re.IGNORECASE) is not None)
        return _result(spec, float(matched), matched, 1)
    lowered = answer.casefold()
    matched = sum(item.casefold() in lowered for item in spec.expected)
    total = len(spec.expected)
    return _result(spec, matched / total, matched, total)


def _result(
    spec: QualitySpec,
    score: float,
    matched: int,
    total: int,
) -> QualityResult:
    return QualityResult(
        rule_id=spec.rule_id,
        scored=True,
        passed=score >= spec.pass_threshold,
        score=score,
        matched=matched,
        total=total,
        reason=f"matched {matched}/{total}",
    )
```

- [ ] **Step 4: Run evaluator tests and static checks**

```bash
uv run pytest tests/test_quality_spec.py -q
uv run ruff check src/agentmesh/eval/quality_spec.py tests/test_quality_spec.py
uv run mypy src/agentmesh/eval/quality_spec.py
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentmesh/eval/quality_spec.py tests/test_quality_spec.py
git commit -m "feat(eval): add evidence-based quality rules"
```

### Task 2: Apply One Rule to Both Benchmark Modes

**Files:**
- Modify: `src/agentmesh/eval/metrics.py`
- Modify: `src/agentmesh/eval/benchmark.py`
- Modify: `src/agentmesh/eval/compare.py`
- Modify: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Write paired quality tests**

Use fake Text/Protocol runners returning `"alpha"` and `"alpha beta"` with:

```yaml
quality:
  rule_id: two-facts-v1
  kind: contains_all
  expected: [alpha, beta]
```

Assert:

```python
assert summary.quality_scored_runs == 1
assert summary.text_quality_mean == 0.5
assert summary.protocol_quality_mean == 1.0
assert summary.text_quality_pass_rate == 0.0
assert summary.protocol_quality_pass_rate == 1.0
assert summary.quality_score_delta == 0.5

rows = read_jsonl(
    paths.benchmark_suite_detail("quality_suite", track="deterministic")
)
assert rows[0]["quality"]["rule_id"] == "two-facts-v1"
assert {row["quality"]["score"] for row in rows} == {0.5, 1.0}
```

Add a second task without `quality` and assert it increments
`quality_unscored_runs` but does not enter quality means.

- [ ] **Step 2: Run the tests and verify missing quality fields**

```bash
uv run pytest tests/test_modes_and_benchmark.py::test_benchmark_applies_same_quality_rule_to_both_modes -q
```

Expected: failure because detail and summary quality evidence are absent.

- [ ] **Step 3: Make runtime quality explicitly optional**

Change `RunMetrics`:

```python
answer_quality_score: float | None = None
quality_rule_id: str = "unscored"
quality_passed: bool | None = None
```

Do not call the old length scorer from Text or Protocol modes. Direct mode runs
are `unscored`; benchmark tasks provide the evidence rule.

Change `CompareSummary.quality_preservation_rate` to `float | None`. Its helper
returns `None` whenever either mode is unscored, so `/compare` displays `N/A`
instead of fabricating equal quality.

- [ ] **Step 4: Add quality to detail records and aggregation**

Add `quality: QualityResult` to `BenchmarkDetailRecord`. Parse task rules using:

```python
raw_quality = task.get("quality")
quality_spec = (
    QualitySpec.model_validate(raw_quality)
    if isinstance(raw_quality, dict)
    else None
)
text_quality = evaluate_quality(text_result.answer, quality_spec)
protocol_quality = evaluate_quality(protocol_result.answer, quality_spec)
```

Store one result with each mode detail row. Aggregate only pairs where
`quality_spec` exists.

Add to `BenchmarkSummary`:

```python
quality_scored_runs: int
quality_unscored_runs: int
text_quality_mean: float | None
protocol_quality_mean: float | None
text_quality_pass_rate: float | None
protocol_quality_pass_rate: float | None
quality_score_delta: float | None
```

Remove `quality_preservation_rate`; schema `2.0` consumers must use explicit
quality fields.

- [ ] **Step 5: Run benchmark and mode tests**

```bash
uv run pytest tests/test_modes_and_benchmark.py tests/test_metrics.py tests/test_compare_prompt.py -q
```

Expected: all selected tests pass with unscored direct mode runs.

- [ ] **Step 6: Commit**

```bash
git add src/agentmesh/eval/metrics.py src/agentmesh/eval/benchmark.py src/agentmesh/eval/compare.py tests/test_modes_and_benchmark.py tests/test_metrics.py tests/test_compare_prompt.py
git commit -m "feat(benchmark): aggregate paired quality evidence"
```

### Task 3: Add Explicit Rules to Contest Suites

**Files:**
- Modify: `examples/benchmarks/continuous_tasks.yaml`
- Modify: `examples/benchmarks/long_context_tasks.yaml`
- Modify: `examples/benchmarks/showcase_benchmark.yaml`
- Modify: task files only when expected outputs need deterministic wording
- Create: `tests/test_benchmark_quality_configs.py`

- [ ] **Step 1: Validate every contest task has a quality rule**

```python
from pathlib import Path

import yaml


def test_all_contest_benchmark_tasks_have_valid_quality_rules() -> None:
    for path in Path("examples/benchmarks").glob("*.yaml"):
        suite = yaml.safe_load(path.read_text(encoding="utf-8"))
        for task in suite["tasks"]:
            spec = QualitySpec.model_validate(task["quality"])
            assert spec.rule_id.startswith(f"{task['id'].lower()}-")
```

- [ ] **Step 2: Run the test and observe missing rules**

```bash
uv run pytest tests/test_benchmark_quality_configs.py -q
```

Expected: failure on the first task without `quality`.

- [ ] **Step 3: Add auditable task rules**

Use `contains_all` for deterministic outputs and `regex` for structural
requirements. Examples:

```yaml
quality:
  rule_id: sa3-output-v1
  kind: contains_all
  expected: ["77", "5"]
```

```yaml
quality:
  rule_id: sa2-codeact-v1
  kind: regex
  pattern: "sandbox exit 0|validated"
```

Rules must check task facts or execution evidence, not answer length, route
names, or generic success prose.

- [ ] **Step 4: Run configuration and offline suite tests**

```bash
uv run pytest tests/test_benchmark_quality_configs.py tests/test_modes_and_benchmark.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/benchmarks tests/test_benchmark_quality_configs.py
git commit -m "test(benchmark): define auditable contest quality rules"
```

### Task 4: Report Quality Evidence Without Breaking AgentShell

**Files:**
- Modify: `src/agentmesh/eval/report.py`
- Modify: `src/agentmesh/eval/dashboard.py`
- Modify: `src/agentmesh/shell/render.py`
- Modify: `tests/test_benchmark_dashboard.py`
- Modify: `tests/test_shell.py`

- [ ] **Step 1: Write report and rendering assertions**

Assert output contains:

```text
Quality scored pairs
Text quality mean / pass rate
Protocol quality mean / pass rate
Quality score delta
Unscored pairs
```

Assert no benchmark UI labels `quality_preservation_rate` as a valid quality
claim.

- [ ] **Step 2: Run UI tests and verify old metric remains**

```bash
uv run pytest tests/test_benchmark_dashboard.py tests/test_shell.py -q
```

Expected: assertions fail until the old display is replaced.

- [ ] **Step 3: Render explicit quality evidence**

Dashboard, report, and shell must show scored/unscored counts before displaying
quality numbers. `None` renders as `N/A`, never zero.

- [ ] **Step 4: Run all UI and AgentShell regression tests**

```bash
uv run pytest tests/test_benchmark_dashboard.py tests/test_shell.py tests/test_compare_prompt.py tests/test_interactive_agent.py -q
```

Expected: all tests pass; command surface, streaming, code fidelity, and error
recovery remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/agentmesh/eval/report.py src/agentmesh/eval/dashboard.py src/agentmesh/shell/render.py tests/test_benchmark_dashboard.py tests/test_shell.py
git commit -m "feat(report): show auditable quality evidence"
```

### Task 5: Verify and Document P2

**Files:**
- Modify: `docs/benchmark_design.md`
- Modify: `docs/progress.md`

- [ ] **Step 1: Document quality schema**

Explain rule kinds, partial credit, pass thresholds, unscored tasks, and why
answer length is prohibited.

- [ ] **Step 2: Run formal gates**

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run agentmesh benchmark --suite standard --no-llm
git diff --check
```

Expected: all commands pass and the standard report contains explicit quality
evidence.

- [ ] **Step 3: Record actual results and commit**

```bash
git add docs/benchmark_design.md docs/progress.md
git commit -m "docs: record evidence-based quality p2"
```

## P2 Exit Gate

- [ ] No production quality metric depends on answer length.
- [ ] Every contest task has a versioned quality rule.
- [ ] Text and Protocol answers are evaluated by the same rule.
- [ ] Unscored tasks are labeled and excluded from means.
- [ ] Quality mean, pass rate, delta, and evidence appear in detail/report/UI.
- [ ] AgentShell interaction regression tests pass.
- [ ] Ruff, mypy, and complete pytest suite pass.
