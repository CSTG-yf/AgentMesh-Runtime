# Benchmark Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a professional, dependency-free HTML dashboard containing every configured and available AgentMesh benchmark result with safe missing-data behavior.

**Architecture:** Add a focused `agentmesh.eval.dashboard` module that discovers configured and generated suites, normalizes CSV/JSONL/report inputs without converting missing values to zero, and renders a self-contained HTML document. Expose generation through the existing Typer CLI and cover parsing, failure isolation, safe serialization, and output completeness with unit tests.

**Tech Stack:** Python 3.11, standard-library CSV/JSON/HTML, PyYAML, Typer, pytest, dependency-free HTML/CSS/JavaScript/SVG.

---

## File Structure

- Create `src/agentmesh/eval/dashboard.py`: artifact discovery, normalization, diagnostics, and self-contained HTML rendering.
- Create `tests/test_benchmark_dashboard.py`: parser, degraded-input, escaping, and complete-output tests.
- Modify `src/agentmesh/cli.py`: add the `dashboard` command.
- Modify `README.md`: document generation and output.

### Task 1: Normalize Complete and Incomplete Benchmark Artifacts

**Files:**
- Create: `tests/test_benchmark_dashboard.py`
- Create: `src/agentmesh/eval/dashboard.py`

- [ ] **Step 1: Write failing discovery and degradation tests**

```python
from pathlib import Path

from agentmesh.eval.dashboard import load_dashboard_data


def test_load_dashboard_data_keeps_configured_and_generated_suites(tmp_path: Path) -> None:
    (tmp_path / "examples/benchmarks").mkdir(parents=True)
    (tmp_path / "examples/benchmarks/one.yaml").write_text(
        "name: configured_suite\nrepeat: 1\ntasks: []\n", encoding="utf-8"
    )
    suite = tmp_path / "runs/latest/benchmarks/generated_suite"
    suite.mkdir(parents=True)
    (suite / "benchmark_summary.csv").write_text(
        "suite_name,total_runs,token_saving_rate\n"
        "generated_suite,2,0.25\n",
        encoding="utf-8",
    )
    (suite / "benchmark_detail.jsonl").write_text(
        '{"task_id":"A1","mode":"text","metrics":{"latency_ms":12}}\n'
        "not-json\n",
        encoding="utf-8",
    )

    data = load_dashboard_data(tmp_path)

    assert [suite["name"] for suite in data["suites"]] == [
        "configured_suite",
        "generated_suite",
    ]
    configured = data["suites"][0]
    assert configured["status"] == "not_generated"
    assert configured["summary"] == {}
    generated = data["suites"][1]
    assert generated["summary"]["token_saving_rate"] == 0.25
    assert generated["details"][0]["metrics"]["latency_ms"] == 12
    assert generated["diagnostics"][0]["kind"] == "malformed_jsonl"
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `uv run pytest tests/test_benchmark_dashboard.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'agentmesh.eval.dashboard'`.

- [ ] **Step 3: Implement discovery and tolerant parsing**

Implement these public boundaries in `src/agentmesh/eval/dashboard.py`:

```python
def load_dashboard_data(root: Path) -> dict[str, object]:
    """Return generated_at, completeness, and normalized suites."""


def _configured_suites(root: Path) -> dict[str, dict[str, object]]:
    """Read YAML suite names and task metadata; isolate malformed files."""


def _generated_suite_dirs(root: Path) -> dict[str, Path]:
    """Return each directory below runs/latest/benchmarks."""


def _read_summary(path: Path, diagnostics: list[dict[str, str]]) -> dict[str, object]:
    """Read the first CSV row and parse finite numeric values."""


def _read_jsonl(path: Path, diagnostics: list[dict[str, str]]) -> list[dict[str, object]]:
    """Keep valid object rows and report each invalid row."""
```

Use `None` for absent values, preserve unknown keys, sort suites by name, and assign
`complete`, `partial`, or `not_generated` from source availability.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_benchmark_dashboard.py -v`

Expected: PASS.

- [ ] **Step 5: Commit normalized artifact loading**

```powershell
git add -- tests/test_benchmark_dashboard.py src/agentmesh/eval/dashboard.py
git commit -m "feat(benchmark): normalize dashboard artifacts"
```

### Task 2: Render a Safe Self-Contained Professional Dashboard

**Files:**
- Modify: `tests/test_benchmark_dashboard.py`
- Modify: `src/agentmesh/eval/dashboard.py`

- [ ] **Step 1: Write failing rendering tests**

```python
from agentmesh.eval.dashboard import generate_dashboard


def test_generate_dashboard_is_self_contained_and_safe(tmp_path: Path) -> None:
    suite = tmp_path / "runs/latest/benchmarks/demo"
    suite.mkdir(parents=True)
    (suite / "benchmark_summary.csv").write_text(
        "suite_name,total_runs,token_saving_rate,custom_metric\n"
        "demo,1,0.5,7\n",
        encoding="utf-8",
    )
    (suite / "benchmark_detail.jsonl").write_text(
        '{"task_id":"</script><b>x</b>","mode":"protocol","metrics":{}}\n',
        encoding="utf-8",
    )

    output = generate_dashboard(tmp_path)
    html = output.read_text(encoding="utf-8")

    assert output.name == "benchmark_dashboard.html"
    assert "AgentMesh Benchmark" in html
    assert "custom_metric" in html
    assert "No benchmark data" in html
    assert "</script><b>x</b>" not in html
    assert "<script src=" not in html
    assert "<link rel=" not in html
```

- [ ] **Step 2: Run the rendering test and verify failure**

Run: `uv run pytest tests/test_benchmark_dashboard.py::test_generate_dashboard_is_self_contained_and_safe -v`

Expected: FAIL because `generate_dashboard` is not defined.

- [ ] **Step 3: Implement HTML serialization and renderer**

Add:

```python
def generate_dashboard(root: Path, output: Path | None = None) -> Path:
    data = load_dashboard_data(root)
    destination = output or root / "runs/latest/benchmarks/benchmark_dashboard.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_html(data), encoding="utf-8")
    return destination


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
```

The `_render_html` template must include:

- responsive navy/blue visual system and tabular numerals;
- global KPI cards with `N/A` and status annotations;
- suite cards and suite selection;
- dependency-free SVG bar comparisons;
- filterable/sortable complete task table;
- grouped catalog generated from every summary key;
- collapsible report and diagnostic evidence;
- explicit empty states and no external URLs.

- [ ] **Step 4: Run dashboard tests**

Run: `uv run pytest tests/test_benchmark_dashboard.py -v`

Expected: PASS.

- [ ] **Step 5: Commit dashboard rendering**

```powershell
git add -- tests/test_benchmark_dashboard.py src/agentmesh/eval/dashboard.py
git commit -m "feat(benchmark): render static results dashboard"
```

### Task 3: Expose, Document, and Verify Dashboard Generation

**Files:**
- Modify: `src/agentmesh/cli.py`
- Modify: `README.md`
- Modify: `tests/test_benchmark_dashboard.py`

- [ ] **Step 1: Write a failing CLI test**

```python
from typer.testing import CliRunner

from agentmesh.cli import app


def test_dashboard_command_generates_html(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["dashboard"])
    assert result.exit_code == 0
    assert (tmp_path / "runs/latest/benchmarks/benchmark_dashboard.html").exists()
```

- [ ] **Step 2: Run the CLI test and verify failure**

Run: `uv run pytest tests/test_benchmark_dashboard.py::test_dashboard_command_generates_html -v`

Expected: FAIL because command `dashboard` does not exist.

- [ ] **Step 3: Add the CLI command**

Add the import and command in `src/agentmesh/cli.py`:

```python
from agentmesh.eval.dashboard import generate_dashboard


@app.command()
def dashboard(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML path; defaults to runs/latest/benchmarks/benchmark_dashboard.html.",
    ),
) -> None:
    path = generate_dashboard(Path.cwd(), output)
    console.print(f"[green]Benchmark dashboard:[/green] {path}")
```

- [ ] **Step 4: Document usage**

Add to `README.md`:

```markdown
### 静态 Benchmark Dashboard

运行 `uv run agentmesh dashboard`，生成
`runs/latest/benchmarks/benchmark_dashboard.html`。页面自动汇总所有 suite，
缺失或损坏的结果显示为 `N/A`/诊断信息，不影响其他数据展示。
```

- [ ] **Step 5: Run full verification**

Run:

```powershell
uv run pytest tests/test_benchmark_dashboard.py -v
uv run ruff check src/agentmesh/eval/dashboard.py src/agentmesh/cli.py tests/test_benchmark_dashboard.py
uv run mypy src/agentmesh/eval/dashboard.py
uv run agentmesh dashboard
```

Expected: all tests, Ruff, and mypy pass; the HTML path is printed.

- [ ] **Step 6: Inspect the generated artifact**

Confirm the document contains all configured suite names, has no `src=` or external
stylesheet references, and opens with populated/empty-state sections without console
errors.

- [ ] **Step 7: Commit CLI and documentation**

```powershell
git add -- src/agentmesh/cli.py README.md tests/test_benchmark_dashboard.py
git commit -m "feat(benchmark): expose static dashboard command"
```
