# P3 LLM Quality-Preserving Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce real-LLM Protocol Mode Agent I/O tokens and latency without lowering its three-repeat quality mean or pass rate.

**Architecture:** Add versioned LLM experiment profiles that isolate baseline and candidate artifacts, then pass an immutable Protocol optimization profile into Protocol Mode. Candidate-only context packing, safe routing, evidence filtering, and concise prompt contracts produce auditable telemetry. A comparison gate validates compatible manifests and refuses to claim efficiency gains until quality constraints pass.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, Typer, orjson, pytest

---

## File Map

- Create `src/agentmesh/eval/llm_experiment.py`: profile schema, profile loading,
  manifest fingerprints, comparison compatibility, and quality guard.
- Create `src/agentmesh/runtime/context_budget.py`: deterministic role-specific
  context packing and audit records.
- Create `src/agentmesh/memory/evidence_pack.py`: relevance filtering,
  deduplication, bounded evidence packing, and telemetry.
- Create `examples/benchmarks/profiles/p3_baseline.yaml`: three-repeat legacy
  baseline profile.
- Create `examples/benchmarks/profiles/p3_candidate.yaml`: three-repeat
  quality-safe candidate profile.
- Modify `src/agentmesh/eval/experiment.py`: include non-secret profile
  provenance in manifest identity.
- Modify `src/agentmesh/llm/client.py`: instrument successful and failed real
  provider calls without exposing secrets.
- Modify `src/agentmesh/eval/benchmark.py`: load a profile, override repeat
  count, isolate artifact variants, fail early when LLM configuration is
  missing, and pass the profile to Protocol Mode.
- Modify `src/agentmesh/storage/paths.py`: profile-scoped benchmark artifact
  directories.
- Modify `src/agentmesh/modes/protocol_mode.py`: apply candidate context,
  route, evidence, and telemetry policies without changing AgentShell defaults.
- Modify `src/agentmesh/modes/text_mode.py`: record real provider call success
  and failure counts for benchmark completeness.
- Modify `src/agentmesh/runtime/decision.py`: explicit quality-safe route
  normalization and persisted reasons.
- Modify `src/agentmesh/eval/metrics.py`: context, routing, and evidence
  telemetry.
- Modify `src/agentmesh/eval/report.py` and
  `src/agentmesh/shell/render.py`: profile provenance and quality-gated
  baseline/candidate comparison.
- Modify `src/agentmesh/cli.py` and `src/agentmesh/shell/session.py`: accept
  `--profile` for benchmark execution and add a comparison command.
- Modify `prompts/planner.md`, `prompts/retriever.md`,
  `prompts/executor.md`, and `prompts/summarizer.md`: concise versioned role
  contracts.
- Create `tests/test_llm_experiment.py`,
  `tests/test_context_budget.py`, `tests/test_evidence_pack.py`,
  `tests/test_cli.py`, and `tests/test_report.py`.
- Modify benchmark, routing, prompt, report, CLI, and AgentShell tests.

### Task 1: Add Reproducible LLM Experiment Profiles and Isolated Artifacts

**Files:**
- Create: `src/agentmesh/eval/llm_experiment.py`
- Create: `examples/benchmarks/profiles/p3_baseline.yaml`
- Create: `examples/benchmarks/profiles/p3_candidate.yaml`
- Create: `tests/test_llm_experiment.py`
- Modify: `src/agentmesh/eval/experiment.py`
- Modify: `src/agentmesh/eval/benchmark.py`
- Modify: `src/agentmesh/storage/paths.py`
- Modify: `src/agentmesh/cli.py`
- Modify: `src/agentmesh/eval/report.py`
- Modify: `src/agentmesh/shell/render.py`
- Modify: `src/agentmesh/llm/client.py`
- Modify: `src/agentmesh/eval/metrics.py`
- Modify: `src/agentmesh/modes/text_mode.py`
- Modify: `src/agentmesh/modes/protocol_mode.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_report.py`
- Modify: `tests/test_experiment_manifest.py`
- Modify: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Write profile and artifact-isolation tests**

```python
from pathlib import Path

import pytest

from agentmesh.eval.llm_experiment import (
    LLMExperimentProfile,
    load_llm_experiment_profile,
)
from agentmesh.storage.paths import RuntimePaths


def test_profile_loads_three_repeat_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.yaml"
    path.write_text(
        """
profile_id: p3-baseline-v1
artifact_label: p3-baseline
repeat: 3
prompt_version: p2
route_policy_version: legacy
optimization_enabled: false
protocol:
  planner_max_chars: 1200
  retriever_max_chars: 1600
  executor_max_chars: 2400
  summarizer_max_chars: 2400
  evidence_max_items: 3
  evidence_min_score: 0.0
""".strip(),
        encoding="utf-8",
    )

    profile = load_llm_experiment_profile(path)

    assert profile.profile_id == "p3-baseline-v1"
    assert profile.repeat == 3
    assert not profile.optimization_enabled


def test_profile_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError):
        LLMExperimentProfile(
            profile_id="bad",
            artifact_label="bad",
            repeat=3,
            protocol={"planner_max_chars": 0},
        )


def test_profile_scopes_benchmark_artifacts(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)

    baseline = paths.benchmark_suite_summary(
        "standard",
        track="llm",
        variant="p3-baseline",
    )
    candidate = paths.benchmark_suite_summary(
        "standard",
        track="llm",
        variant="p3-candidate",
    )

    assert baseline != candidate
    assert baseline.as_posix().endswith(
        "benchmarks/standard/llm/p3-baseline/benchmark_summary.csv"
    )


def test_instrumented_llm_records_provider_failure() -> None:
    stats = LLMCallStats()
    client = InstrumentedLLMClient(FailingLLM(), stats)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        client.complete(agent_name="planner", messages=[])

    assert stats.call_count == 1
    assert stats.error_count == 1
```

Define the test double in the same test file:

```python
class FailingLLM:
    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        del agent_name, messages, variables
        raise RuntimeError("provider unavailable")
```

Create `tests/test_cli.py` with:

```python
from pathlib import Path

from typer.testing import CliRunner

from agentmesh.cli import app


def test_config_command_never_prints_api_key(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AGENTMESH_LLM_BASE_URL=https://llm.example/v1",
                "AGENTMESH_LLM_API_KEY=secret-value",
                "AGENTMESH_LLM_MODEL=test-model",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["config", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert "secret-value" not in result.stdout
    assert '"llm_api_key": "configured"' in result.stdout
```

Create `tests/test_report.py` with:

```python
from pathlib import Path

from agentmesh.eval.report import generate_report
from agentmesh.storage.paths import RuntimePaths


def test_profiled_report_stays_in_variant_directory(tmp_path: Path) -> None:
    paths = RuntimePaths(root=tmp_path)
    summary = paths.benchmark_suite_summary(
        "suite",
        track="llm",
        variant="p3-baseline",
    )
    summary.parent.mkdir(parents=True)
    summary.write_text("suite_name,total_runs\nsuite,1\n", encoding="utf-8")

    report = generate_report(
        paths,
        suite_name="suite",
        track="llm",
        variant="p3-baseline",
    )

    assert report.parent == summary.parent
```

- [ ] **Step 2: Run the tests and verify missing APIs**

Run:

```bash
uv run pytest tests/test_llm_experiment.py -q
```

Expected: collection fails because `llm_experiment` and `variant` support do
not exist.

- [ ] **Step 3: Implement the immutable profile schema**

Create `src/agentmesh/eval/llm_experiment.py` with:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProtocolOptimizationProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    planner_max_chars: int = Field(default=1200, gt=0)
    retriever_max_chars: int = Field(default=1600, gt=0)
    executor_max_chars: int = Field(default=2400, gt=0)
    summarizer_max_chars: int = Field(default=2400, gt=0)
    evidence_max_items: int = Field(default=3, gt=0)
    evidence_min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMExperimentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    artifact_label: str
    repeat: int = Field(default=3, gt=0)
    prompt_version: str = "p2"
    route_policy_version: Literal["legacy", "quality_safe_v1"] = "legacy"
    optimization_enabled: bool = False
    protocol: ProtocolOptimizationProfile = Field(
        default_factory=ProtocolOptimizationProfile
    )

    @property
    def sha256(self) -> str:
        payload = orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(payload).hexdigest()


def load_llm_experiment_profile(path: Path) -> LLMExperimentProfile:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("LLM experiment profile must be a mapping")
    return LLMExperimentProfile.model_validate(loaded)
```

Create the baseline profile using the exact values from the first test. Create
the candidate profile with:

```yaml
profile_id: p3-candidate-v1
artifact_label: p3-candidate
repeat: 3
prompt_version: p3-v1
route_policy_version: quality_safe_v1
optimization_enabled: true
protocol:
  planner_max_chars: 700
  retriever_max_chars: 900
  executor_max_chars: 1400
  summarizer_max_chars: 1500
  evidence_max_items: 2
  evidence_min_score: 0.35
```

- [ ] **Step 4: Add `variant` to benchmark paths**

Change all `benchmark_suite_*` helpers to accept `variant: str | None = None`.
The directory rule must be:

```python
def benchmark_suite_dir(
    self,
    suite_name: str,
    track: str | None = None,
    variant: str | None = None,
) -> Path:
    path = self.benchmark_dir / _slugify_path_name(suite_name)
    if track:
        path /= _slugify_path_name(track)
    if variant:
        path /= _slugify_path_name(variant)
    return path
```

Existing calls without `variant` must retain their current locations.

- [ ] **Step 5: Add profile provenance to the manifest**

Add these frozen fields to `ExperimentManifest`:

```python
profile_id: str = "default"
profile_sha256: str = ""
prompt_version: str = "default"
route_policy_version: str = "default"
model_fingerprint: str = ""
prompt_tree_sha256: str = ""
quality_rules_sha256: str = ""
```

Extend `build_experiment_manifest` with matching keyword arguments and include
all seven values in the experiment identity. Compute `model_fingerprint` as:

```python
def model_fingerprint(base_url: str, model: str) -> str:
    host_and_model = f"{base_url.rstrip('/')}|{model}"
    return hashlib.sha256(host_and_model.encode()).hexdigest()[:16]
```

Never include an API key, authorization header, or raw secret in a manifest.

- [ ] **Step 6: Instrument provider calls and incomplete samples**

Add to `src/agentmesh/llm/client.py`:

```python
class LLMCallStats(BaseModel):
    call_count: int = 0
    error_count: int = 0


class InstrumentedLLMClient:
    def __init__(self, client: LLMClient, stats: LLMCallStats) -> None:
        self._client = client
        self._stats = stats

    def complete(
        self,
        *,
        agent_name: str,
        messages: list[ChatMessage],
        variables: dict[str, object] | None = None,
    ) -> str:
        self._stats.call_count += 1
        try:
            return self._client.complete(
                agent_name=agent_name,
                messages=messages,
                variables=variables,
            )
        except Exception:
            self._stats.error_count += 1
            raise
```

Add `llm_call_count` and `llm_error_count` to `RunMetrics`. Wrap the resolved
LLM client in Text and Protocol modes and copy the final counters to metrics.
Profile-free AgentShell runs may still use existing fallback behavior. During
a profiled benchmark, any Text or Protocol result with
`llm_error_count > 0` raises `BenchmarkConfigError` and the pair is not
appended to detail output.

- [ ] **Step 7: Wire profiles into benchmark execution**

Extend:

```python
def run_benchmark(
    suite_path: Path,
    paths: RuntimePaths,
    *,
    use_llm: bool = True,
    profile: LLMExperimentProfile | None = None,
    progress_callback: Callable[[BenchmarkProgressEvent], None] | None = None,
) -> BenchmarkSummary:
```

Use `profile.repeat` when a profile is supplied. If `use_llm` is true and the
project LLM configuration is incomplete, raise:

```python
BenchmarkConfigError(
    "Real LLM benchmark requires base_url, api_key, and model"
)
```

Pass the profile to `run_protocol_mode(..., experiment_profile=profile)`.
Task 1 must add this optional argument through `ProtocolRunSession` and
`_run_protocol_mode_impl` but must not apply candidate optimization yet.
Use `profile.artifact_label` as the path variant for summary, detail, report,
and manifest. Preserve existing behavior when `profile is None`.

Compute `quality_rules_sha256` from the canonical sorted JSON representation
of every task ID and `quality` mapping. Compute `prompt_tree_sha256` from the
active prompt directory. Pass both hashes into the manifest.

Add `artifact_variant: str = ""` and `quality_rules_sha256: str = ""` to
`BenchmarkSummary`. Pass `summary.artifact_variant or None` through
`generate_report`, `render_benchmark_artifacts`, and every profiled CLI path so
the report is written beside the matching summary rather than into the legacy
LLM directory.

Add a CLI `config` command that prints only:

```python
{
    "llm_configured": config.llm.configured,
    "llm_base_url": config.llm.base_url or "",
    "llm_model": config.llm.model or "",
    "llm_api_key": "configured" if config.llm.api_key else "missing",
}
```

The command must never serialize or print the secret value.

Add `--profile PATH` to the CLI benchmark command. Reject `--no-llm` combined
with `--profile`, because P3 profiles are real-LLM evidence only.

- [ ] **Step 8: Run profile, manifest, CLI, and compatibility tests**

```bash
uv run pytest tests/test_llm_experiment.py tests/test_experiment_manifest.py tests/test_modes_and_benchmark.py tests/test_cli.py tests/test_report.py -q
uv run ruff check src/agentmesh/eval/llm_experiment.py src/agentmesh/eval/experiment.py src/agentmesh/eval/benchmark.py src/agentmesh/storage/paths.py src/agentmesh/llm/client.py
uv run mypy src
```

Expected: all commands pass and legacy benchmark paths remain unchanged.

- [ ] **Step 9: Commit profile infrastructure**

```bash
git add src/agentmesh/eval/llm_experiment.py src/agentmesh/eval/experiment.py src/agentmesh/eval/benchmark.py src/agentmesh/eval/report.py src/agentmesh/storage/paths.py src/agentmesh/cli.py src/agentmesh/llm/client.py src/agentmesh/eval/metrics.py src/agentmesh/modes/text_mode.py src/agentmesh/modes/protocol_mode.py src/agentmesh/shell/render.py examples/benchmarks/profiles tests/test_llm_experiment.py tests/test_experiment_manifest.py tests/test_modes_and_benchmark.py tests/test_cli.py tests/test_report.py
git commit -m "feat(benchmark): add reproducible llm experiment profiles"
```

- [ ] **Step 10: Run and preserve the real LLM baseline before optimization**

First print only masked configuration:

```bash
uv run agentmesh config
```

Then run:

```bash
uv run agentmesh benchmark --suite standard --profile examples/benchmarks/profiles/p3_baseline.yaml
```

Expected:

- 18 paired samples complete under
  `runs/latest/benchmarks/continuous_tasks/llm/p3-baseline/`;
- manifest says `profile_id=p3-baseline-v1` and `repeat_count=3`;
- every task is quality-scored;
- no provider failure is converted into a completed zero score.

Record the baseline manifest hash, Protocol quality mean/pass rate, Protocol
Agent I/O token mean, and latency P50/P95 in a new uncommitted
`docs/p3_llm_results.md`. Do not change prompts or optimization behavior before
this baseline exists.

### Task 2: Add Role-Specific Context Budgets and Audit Telemetry

**Files:**
- Create: `src/agentmesh/runtime/context_budget.py`
- Create: `tests/test_context_budget.py`
- Modify: `src/agentmesh/eval/metrics.py`
- Modify: `src/agentmesh/modes/protocol_mode.py`
- Modify: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Write context packing tests**

```python
from agentmesh.runtime.context_budget import ContextPart, pack_context


def test_context_packer_keeps_required_parts_before_evidence() -> None:
    result = pack_context(
        role="summarizer",
        max_chars=45,
        parts=[
            ContextPart(name="task", text="Return sum 77 and even count 5.", required=True),
            ContextPart(name="execution", text="sandbox exit 0", required=True),
            ContextPart(name="evidence", text="x" * 200, required=False),
        ],
    )

    assert "sum 77" in result.text
    assert "sandbox exit 0" in result.text
    assert "x" * 50 not in result.text
    assert result.audit.original_chars > result.audit.retained_chars
    assert result.audit.truncated


def test_required_content_over_budget_uses_safe_fallback() -> None:
    result = pack_context(
        role="executor",
        max_chars=5,
        parts=[ContextPart(name="task", text="required task", required=True)],
    )

    assert result.text == "required task"
    assert result.audit.safe_fallback
```

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
uv run pytest tests/test_context_budget.py -q
```

Expected: collection fails because `context_budget` does not exist.

- [ ] **Step 3: Implement the deterministic packer**

Create immutable `ContextPart`, `ContextAudit`, and `PackedContext` Pydantic
models. Implement:

```python
def pack_context(
    *,
    role: str,
    max_chars: int,
    parts: list[ContextPart],
) -> PackedContext:
    required = [part for part in parts if part.required and part.text.strip()]
    optional = [part for part in parts if not part.required and part.text.strip()]
    required_text = "\n\n".join(part.text.strip() for part in required)
    original_chars = sum(len(part.text) for part in required + optional)
    if len(required_text) > max_chars:
        return PackedContext(
            text=required_text,
            audit=ContextAudit(
                role=role,
                max_chars=max_chars,
                original_chars=original_chars,
                retained_chars=len(required_text),
                truncated=False,
                safe_fallback=True,
                included_parts=[part.name for part in required],
                dropped_parts=[part.name for part in optional],
            ),
        )
    retained = list(required)
    current = len(required_text)
    for part in optional:
        separator = 2 if retained else 0
        available = max_chars - current - separator
        if available <= 0:
            break
        retained.append(
            part if len(part.text) <= available
            else part.model_copy(update={"text": part.text[:available]})
        )
        current += separator + min(len(part.text), available)
    text = "\n\n".join(part.text.strip() for part in retained)
    included = {part.name for part in retained}
    return PackedContext(
        text=text,
        audit=ContextAudit(
            role=role,
            max_chars=max_chars,
            original_chars=original_chars,
            retained_chars=len(text),
            truncated=len(text) < original_chars,
            safe_fallback=False,
            included_parts=[part.name for part in retained],
            dropped_parts=[
                part.name for part in required + optional if part.name not in included
            ],
        ),
    )
```

- [ ] **Step 4: Apply packing only to candidate Protocol runs**

Extend `run_protocol_mode`, `ProtocolRunSession`, and
`_run_protocol_mode_impl` with:

```python
experiment_profile: LLMExperimentProfile | None = None
```

When `profile.optimization_enabled` is false or no profile is supplied, retain
the existing inputs exactly. For candidate runs, pack:

- Planner: required task.
- Retriever: required task and planner intent.
- Executor: required task and optional evidence digest.
- Summarizer: required task and execution result, optional evidence digest.

Persist each `ContextAudit` in `RunMetrics.context_audits`:

```python
context_audits: list[dict[str, object]] = Field(default_factory=list)
context_original_chars: int = 0
context_retained_chars: int = 0
context_safe_fallback_count: int = 0
```

Do not truncate the task or execution result if doing so would remove required
quality facts.

- [ ] **Step 5: Run context and Protocol regressions**

```bash
uv run pytest tests/test_context_budget.py tests/test_modes_and_benchmark.py tests/test_real_agent_protocol_mode.py tests/test_shell.py -q
uv run ruff check src/agentmesh/runtime/context_budget.py src/agentmesh/modes/protocol_mode.py
uv run mypy src
```

Expected: all tests pass; AgentShell runs without a profile retain existing
behavior.

- [ ] **Step 6: Commit context budgets**

```bash
git add src/agentmesh/runtime/context_budget.py src/agentmesh/eval/metrics.py src/agentmesh/modes/protocol_mode.py tests/test_context_budget.py tests/test_modes_and_benchmark.py
git commit -m "feat(protocol): add auditable role context budgets"
```

### Task 3: Add Quality-Safe Routing and Compact Memory Evidence

**Files:**
- Create: `src/agentmesh/memory/evidence_pack.py`
- Create: `tests/test_evidence_pack.py`
- Modify: `src/agentmesh/runtime/decision.py`
- Modify: `src/agentmesh/modes/protocol_mode.py`
- Modify: `src/agentmesh/eval/metrics.py`
- Modify: `tests/test_dynamic_agent_routing.py`
- Modify: `tests/test_modes_and_benchmark.py`

- [ ] **Step 1: Write route and evidence tests**

```python
from agentmesh.memory.evidence_pack import pack_evidence
from agentmesh.runtime.decision import PlannerDecision


def test_candidate_analysis_without_reuse_skips_retriever() -> None:
    decision = PlannerDecision.from_task(
        "Explain why quicksort can degrade on a partially sorted array."
    ).for_policy("quality_safe_v1")

    assert decision.execution_route == ["planner", "summarizer"]
    assert decision.reason.endswith("quality_safe_v1: retrieval not required")


def test_candidate_code_task_keeps_executor() -> None:
    decision = PlannerDecision.from_task(
        "Implement quicksort, run it, and print the sorted result."
    ).for_policy("quality_safe_v1")

    assert "executor" in decision.execution_route
    assert decision.execution_route[-1] == "summarizer"


def test_evidence_pack_filters_deduplicates_and_bounds() -> None:
    packed = pack_evidence(
        [
            {"memory_id": "m1", "title": "A", "snippet": "alpha", "score": 0.9},
            {"memory_id": "m1", "title": "A2", "snippet": "alpha", "score": 0.8},
            {"memory_id": "m2", "title": "B", "snippet": "beta", "score": 0.2},
            {"memory_id": "m3", "title": "C", "snippet": "gamma", "score": 0.7},
        ],
        min_score=0.35,
        max_items=2,
        max_chars=80,
    )

    assert packed.accepted_ids == ["m1", "m3"]
    assert packed.audit.candidate_count == 4
    assert packed.audit.accepted_count == 2
    assert packed.audit.deduplicated_count == 1
    assert packed.audit.filtered_count == 1
    assert len(packed.digest) <= 80
```

- [ ] **Step 2: Run tests and verify missing policy/packer**

```bash
uv run pytest tests/test_evidence_pack.py tests/test_dynamic_agent_routing.py -q
```

Expected: the new tests fail because `for_policy` and `pack_evidence` are
missing.

- [ ] **Step 3: Implement evidence packing**

Create frozen `EvidencePackAudit` and `PackedEvidence` models. `pack_evidence`
must:

1. read numeric `score`, defaulting to `0.0`;
2. sort descending by score;
3. reject scores below `min_score`;
4. deduplicate first by non-empty `memory_id`, otherwise by normalized
   `title|snippet`;
5. retain at most `max_items`;
6. format each line as
   `"[<memory_id>] <title> (score=<score:.3f>): <snippet>"`;
7. stop before `max_chars`;
8. return counts and injected UTF-8 bytes.

- [ ] **Step 4: Implement explicit candidate route policy**

Add:

```python
def for_policy(self, policy_version: str) -> PlannerDecision:
    if policy_version != "quality_safe_v1":
        return self
    if self.need_tool_execution:
        return self
    if self.task_type in {
        "chat",
        "summary_only",
        "calculation",
        "analysis_or_report",
    }:
        return self.model_copy(
            update={
                "need_retrieval": False,
                "execution_route": [
                    agent
                    for agent in self.execution_route
                    if agent != "retriever"
                ],
                "required_capabilities": [
                    capability
                    for capability in self.required_capabilities
                    if capability not in {"memory.semantic_search", "evidence.refine"}
                ],
                "reason": f"{self.reason}; quality_safe_v1: retrieval not required",
            }
        ).normalized()
    return self
```

Apply the policy after `PlannerDecision.from_llm_or_task`. Ambiguous `general`,
explicit memory/reuse tasks, code review, knowledge lookup, benchmark, and
validation retain their full safe routes.

- [ ] **Step 5: Integrate packed evidence and telemetry**

For optimized profiles, replace `_evidence_digest` input to Executor and
Summarizer with `pack_evidence(...)`. Add:

```python
route_policy_version: str = "default"
route_reason: str = ""
evidence_candidate_count: int = 0
evidence_accepted_count: int = 0
evidence_deduplicated_count: int = 0
evidence_filtered_count: int = 0
```

to `RunMetrics`. Continue recording `memory_evidence_bytes` as bytes actually
injected, not bytes retrieved.

- [ ] **Step 6: Run routing, memory, mode, and shell tests**

```bash
uv run pytest tests/test_evidence_pack.py tests/test_dynamic_agent_routing.py tests/test_agent_state_refs.py tests/test_hybrid_memory.py tests/test_modes_and_benchmark.py tests/test_shell.py -q
uv run ruff check src/agentmesh/memory/evidence_pack.py src/agentmesh/runtime/decision.py src/agentmesh/modes/protocol_mode.py
uv run mypy src
```

Expected: all tests pass; profile-free AgentShell routing remains unchanged.

- [ ] **Step 7: Commit route and evidence optimization**

```bash
git add src/agentmesh/memory/evidence_pack.py src/agentmesh/runtime/decision.py src/agentmesh/modes/protocol_mode.py src/agentmesh/eval/metrics.py tests/test_evidence_pack.py tests/test_dynamic_agent_routing.py tests/test_modes_and_benchmark.py
git commit -m "feat(protocol): add quality-safe routing and evidence packing"
```

### Task 4: Tighten Prompt Contracts and Record Prompt Provenance

**Files:**
- Modify: `prompts/planner.md`
- Modify: `prompts/retriever.md`
- Modify: `prompts/executor.md`
- Modify: `prompts/summarizer.md`
- Modify: `src/agentmesh/prompts/store.py`
- Modify: `src/agentmesh/eval/experiment.py`
- Modify: `tests/test_prompt_and_llm.py`
- Modify: `tests/test_experiment_manifest.py`

- [ ] **Step 1: Write prompt contract tests**

Add:

```python
def test_p3_prompts_are_concise_role_contracts(tmp_path: Path) -> None:
    store = PromptTemplateStore(prompt_dir=Path("prompts"))

    planner = store.render("planner", {"input": "task"})
    retriever = store.render("retriever", {"input": "task"})
    executor = store.render("executor", {"input": "task"})
    summarizer = store.render("summarizer", {"input": "task"})

    assert "Do not restate the task" in planner
    assert "Return compact evidence facts" in retriever
    assert "Return only executable code or structured validation" in executor
    assert "Answer the user directly" in summarizer
    assert "required facts" in summarizer
```

Add a prompt fingerprint test:

```python
def test_prompt_fingerprint_changes_with_prompt_content(tmp_path: Path) -> None:
    prompt = tmp_path / "planner.md"
    prompt.write_text("version one", encoding="utf-8")
    first = prompt_tree_sha256(tmp_path)
    prompt.write_text("version two", encoding="utf-8")
    assert prompt_tree_sha256(tmp_path) != first
```

- [ ] **Step 2: Run tests and verify missing contracts/fingerprint**

```bash
uv run pytest tests/test_prompt_and_llm.py tests/test_experiment_manifest.py -q
```

Expected: new assertions fail.

- [ ] **Step 3: Replace prompts with concise contracts**

Each prompt must include its current persona and these exact candidate
contracts:

```text
Planner: Return intent, required capabilities, route, and one short reason.
Do not restate the task.

Retriever: Return compact evidence facts with source identity and relevance.
Do not add general commentary.

Executor: Return only executable code or structured validation observations.
Do not write the final user answer.

Summarizer: Answer the user directly. Preserve all required facts, computed
outputs, and validated execution evidence. Do not narrate the agent route.
```

Do not add task-specific answers to prompts.

- [ ] **Step 4: Add prompt-tree fingerprint**

Implement:

```python
def prompt_tree_sha256(prompt_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(prompt_dir.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
```

Record this hash in the manifest. The profile's human-readable
`prompt_version` and the content hash must both be present.

- [ ] **Step 5: Run prompt, manifest, and real-agent contract tests**

```bash
uv run pytest tests/test_prompt_and_llm.py tests/test_experiment_manifest.py tests/test_real_agent_protocol_mode.py tests/test_compare_prompt.py -q
uv run ruff check prompts src/agentmesh/prompts/store.py src/agentmesh/eval/experiment.py
uv run mypy src
```

Expected: all tests pass.

- [ ] **Step 6: Commit prompt contracts**

```bash
git add prompts src/agentmesh/prompts/store.py src/agentmesh/eval/experiment.py tests/test_prompt_and_llm.py tests/test_experiment_manifest.py
git commit -m "feat(prompts): add concise p3 agent contracts"
```

### Task 5: Add Quality-Gated Baseline/Candidate Comparison

**Files:**
- Modify: `src/agentmesh/eval/llm_experiment.py`
- Create: `src/agentmesh/eval/llm_comparison.py`
- Create: `tests/test_llm_comparison.py`
- Modify: `src/agentmesh/cli.py`
- Modify: `src/agentmesh/eval/report.py`
- Modify: `src/agentmesh/shell/render.py`
- Modify: `src/agentmesh/shell/session.py`
- Modify: `tests/test_shell.py`

- [ ] **Step 1: Write compatibility and quality guard tests**

```python
from agentmesh.eval.llm_comparison import compare_llm_experiments


def test_comparison_rejects_incompatible_model_fingerprints() -> None:
    with pytest.raises(ValueError, match="model fingerprint"):
        compare_llm_experiments(
            baseline=_experiment(model_fingerprint="model-a"),
            candidate=_experiment(model_fingerprint="model-b"),
        )


def test_quality_guard_blocks_efficiency_claim_when_quality_drops() -> None:
    result = compare_llm_experiments(
        baseline=_experiment(
            protocol_quality_mean=0.9,
            protocol_quality_pass_rate=0.8,
        ),
        candidate=_experiment(
            protocol_quality_mean=0.8,
            protocol_quality_pass_rate=0.8,
            protocol_token_mean=50,
            protocol_latency_p50=100,
            protocol_latency_p95=200,
        ),
    )

    assert not result.quality_gate_passed
    assert not result.efficiency_claim_allowed
    assert "quality mean regressed" in result.reasons


def test_quality_guard_accepts_lower_tokens_and_latency() -> None:
    result = compare_llm_experiments(
        baseline=_experiment(
            protocol_quality_mean=0.9,
            protocol_quality_pass_rate=0.8,
            protocol_token_mean=100,
            protocol_latency_p50=200,
            protocol_latency_p95=400,
        ),
        candidate=_experiment(
            text_quality_mean=0.9,
            text_quality_pass_rate=0.8,
            protocol_quality_mean=0.9,
            protocol_quality_pass_rate=0.8,
            protocol_token_mean=70,
            protocol_latency_p50=150,
            protocol_latency_p95=390,
        ),
    )

    assert result.quality_gate_passed
    assert result.efficiency_gate_passed
    assert result.efficiency_claim_allowed
```

The `_experiment` test helper must construct complete typed inputs, not raw
unvalidated dictionaries.

- [ ] **Step 2: Run tests and verify comparison is missing**

```bash
uv run pytest tests/test_llm_comparison.py -q
```

Expected: collection fails because `llm_comparison` does not exist.

- [ ] **Step 3: Implement typed comparison**

Create:

```python
class LLMExperimentResult(BaseModel):
    profile_id: str
    suite_sha256: str
    model_fingerprint: str
    quality_rules_sha256: str
    repeat_count: int
    text_quality_mean: float
    text_quality_pass_rate: float
    protocol_quality_mean: float
    protocol_quality_pass_rate: float
    protocol_token_mean: float
    protocol_latency_p50: float
    protocol_latency_p95: float


class LLMComparisonResult(BaseModel):
    compatible: bool
    quality_gate_passed: bool
    efficiency_gate_passed: bool
    efficiency_claim_allowed: bool
    protocol_quality_mean_delta: float
    protocol_quality_pass_rate_delta: float
    protocol_token_reduction_rate: float | None
    protocol_latency_p50_reduction_rate: float | None
    protocol_latency_p95_reduction_rate: float | None
    reasons: list[str]
```

`compare_llm_experiments` must reject mismatched suite, model, quality-rule
hash, or repeat count. The quality gate is:

```python
quality_ok = (
    candidate.protocol_quality_mean >= baseline.protocol_quality_mean
    and candidate.protocol_quality_pass_rate >= baseline.protocol_quality_pass_rate
    and candidate.protocol_quality_mean >= candidate.text_quality_mean - 0.05
    and candidate.protocol_quality_pass_rate
    >= candidate.text_quality_pass_rate - (1 / 18)
)
```

The efficiency gate requires lower Protocol token mean, lower P50 or P95, and
no more than 10% regression in the other percentile.

- [ ] **Step 4: Add CLI and AgentShell comparison surface**

Add:

```bash
agentmesh benchmark-compare \
  --suite continuous_tasks \
  --baseline p3-baseline \
  --candidate p3-candidate
```

and:

```text
/benchmark-compare continuous_tasks p3-baseline p3-candidate
```

Render compatibility, quality gate, efficiency gate, each delta, and reasons.
Never print "improved" when `efficiency_claim_allowed` is false.

- [ ] **Step 5: Run comparison, report, and AgentShell tests**

```bash
uv run pytest tests/test_llm_comparison.py tests/test_shell.py tests/test_benchmark_dashboard.py tests/test_modes_and_benchmark.py -q
uv run ruff check src/agentmesh/eval/llm_comparison.py src/agentmesh/cli.py src/agentmesh/shell
uv run mypy src
```

Expected: all tests pass and existing AgentShell commands are unchanged.

- [ ] **Step 6: Commit the quality gate**

```bash
git add src/agentmesh/eval/llm_experiment.py src/agentmesh/eval/llm_comparison.py src/agentmesh/cli.py src/agentmesh/eval/report.py src/agentmesh/shell tests/test_llm_comparison.py tests/test_shell.py
git commit -m "feat(benchmark): gate llm efficiency claims on quality"
```

### Task 6: Run Candidate Experiment and Evaluate the P3 Gate

**Files:**
- Modify: `docs/p3_llm_results.md`

- [ ] **Step 1: Verify baseline artifacts still exist**

```bash
uv run agentmesh benchmark-compare \
  --suite continuous_tasks \
  --baseline p3-baseline \
  --candidate p3-candidate
```

Expected before the candidate run: a clear "candidate artifacts not found"
error. Baseline artifacts and manifest must remain intact.

- [ ] **Step 2: Run the three-repeat real LLM candidate**

```bash
uv run agentmesh benchmark --suite standard --profile examples/benchmarks/profiles/p3_candidate.yaml
```

Expected: 18 paired samples complete under
`runs/latest/benchmarks/continuous_tasks/llm/p3-candidate/`.

If any provider call fails, stop. Do not compare an incomplete candidate with
the baseline.

- [ ] **Step 3: Execute the quality-first comparison**

```bash
uv run agentmesh benchmark-compare \
  --suite continuous_tasks \
  --baseline p3-baseline \
  --candidate p3-candidate
```

Expected: output states whether compatibility, quality, and efficiency gates
pass and lists measured deltas.

- [ ] **Step 4: Handle a failed quality gate**

If quality mean or pass rate regresses:

1. inspect failed task detail rows and final answers;
2. loosen only the responsible context budget or restore required evidence;
3. add a deterministic regression test for the lost fact;
4. increment candidate profile ID and artifact label;
5. rerun all 18 candidate samples;
6. never reuse partial or failed candidate artifacts.

Do not weaken P2 quality rules to make the gate pass.

- [ ] **Step 5: Record exact evidence**

Update `docs/p3_llm_results.md` with:

- baseline and candidate experiment IDs and manifest hashes;
- model fingerprint and prompt-tree hashes;
- quality means and pass rates for Text and Protocol;
- Protocol token mean and reduction rate;
- Protocol latency P50/P95 and reduction rates;
- context retained/original characters and safe fallback count;
- route counts and evidence filtering counts;
- final gate status and any failed-task analysis.

- [ ] **Step 6: Commit measured P3 results**

```bash
git add docs/p3_llm_results.md examples/benchmarks/profiles/p3_candidate.yaml
git commit -m "docs: record quality-gated llm efficiency p3"
```

### Task 7: Full Verification, Documentation, and Publication

**Files:**
- Modify: `docs/benchmark_design.md`
- Modify: `docs/progress.md`
- Modify: `README.md`

- [ ] **Step 1: Document P3 semantics**

Document:

- real LLM as the contest-quality track;
- deterministic mode as CI/fallback only;
- profile-scoped artifacts and manifest compatibility;
- context budgets, safe fallback, route reasons, and evidence audit;
- quality-first acceptance gate;
- the exact baseline/candidate commands.

- [ ] **Step 2: Run formal gates**

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run agentmesh benchmark-compare --suite continuous_tasks --baseline p3-baseline --candidate p3-candidate
git diff --check
```

Expected:

- Ruff passes;
- mypy reports no issues;
- complete pytest passes with only environment-dependent skips;
- comparison reads complete three-repeat experiments;
- quality gate passes before any efficiency claim;
- Git diff has no whitespace errors.

- [ ] **Step 3: Verify P3 exit criteria directly**

```bash
uv run pytest \
  tests/test_llm_experiment.py \
  tests/test_context_budget.py \
  tests/test_evidence_pack.py \
  tests/test_llm_comparison.py \
  tests/test_dynamic_agent_routing.py \
  tests/test_prompt_and_llm.py \
  tests/test_shell.py -q
```

Expected: all dedicated P3 and AgentShell regression tests pass.

- [ ] **Step 4: Record final results and commit**

```bash
git add README.md docs/benchmark_design.md docs/progress.md
git commit -m "docs: complete quality-preserving llm efficiency p3"
```

- [ ] **Step 5: Push `master` without rewriting the preserved branch**

```bash
git fetch agentbus master
git merge-base --is-ancestor agentbus/master master
git push agentbus master
```

Expected: fast-forward push succeeds. Verify
`codex/agentmesh-runtime-mvp` remains at `3e8fbaf`.

## P3 Exit Gate

- [ ] Baseline was captured before candidate optimization.
- [ ] Baseline and candidate each contain 18 complete real-LLM paired samples.
- [ ] Model, suite, repeat, and quality-rule fingerprints are compatible.
- [ ] Protocol quality mean and pass rate do not regress from baseline.
- [ ] Protocol quality remains within the approved Text tolerances.
- [ ] Protocol mean Agent I/O tokens decrease.
- [ ] Protocol latency P50 or P95 decreases; the other regresses by at most 10%.
- [ ] Context compaction, route reasons, and evidence filtering are auditable.
- [ ] Deterministic mode remains a CI/fallback gate, not a quality claim.
- [ ] AgentShell behavior and regression tests remain intact.
- [ ] Ruff, mypy, complete pytest, and P3 dedicated tests pass.
