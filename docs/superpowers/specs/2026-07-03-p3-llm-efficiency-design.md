# P3 LLM Quality-Preserving Efficiency Design

## Objective

Optimize the real LLM benchmark track so Protocol Mode uses fewer Agent I/O
tokens and has lower latency without reducing answer quality. The deterministic
track remains a CI, protocol, fallback, and regression gate; it is not an
intelligence-quality target.

## Acceptance Policy

P3 uses the standard contest suite for one baseline experiment and one
post-optimization experiment. Each experiment runs three paired repeats with
the same:

- LLM provider, model, endpoint, and model parameters;
- task inputs and versioned quality rules;
- suite seed and alternating Text/Protocol execution order;
- runtime environment and benchmark schema.

The optimized Protocol result is accepted only when:

1. Protocol quality mean is not lower than its three-repeat baseline.
2. Protocol quality pass rate is not lower than its three-repeat baseline.
3. Candidate Protocol quality mean is no more than `0.05` below candidate Text,
   and Protocol pass rate is no more than one sample (`1/18`) below Text.
4. Protocol mean Agent I/O tokens decrease, and at least one of latency P50 or
   P95 decreases.
5. The other latency percentile does not regress by more than `10%`.

Raw artifacts, manifests, detail rows, and summaries from both experiments
must remain available for audit. A single run is not sufficient evidence.

## Architecture

### 1. LLM Experiment Profile

Add an explicit benchmark experiment profile for real LLM runs. It records the
quality guard, repeat count, context budgets, route policy version, prompt
version, and a non-secret model fingerprint in the experiment manifest.

Baseline and candidate runs use separate artifact identities so the candidate
cannot overwrite the baseline. The comparison layer validates that the suite,
model fingerprint, repeat count, and quality-rule set are compatible before
reporting a P3 result.

### 2. Stage-Specific Context Budgets

Protocol Mode will stop treating all state as equally valuable. Each LLM stage
receives a bounded input assembled for its role:

- Planner: task text plus compact routing constraints.
- Retriever: planner intent, retrieval query, and compact task facts.
- Executor: task, executable requirements, and only relevant retrieved
  evidence.
- Summarizer: task, validated execution result, compact evidence digest, and
  required answer format.

Budgeting is deterministic, preserves complete task instructions, and trims
low-priority evidence before required facts or execution output. Every
compaction records original characters, retained characters, and truncation
status for audit.

### 3. Quality-Safe Dynamic Route

The planner continues to choose Agent capabilities, but route normalization
applies explicit safe rules:

- Retrieval is skipped when the task has no memory dependency or reuse intent.
- Executor is skipped only when the task requires neither computation, code,
  validation, nor tool execution.
- Summarizer remains responsible for the final answer.
- Any ambiguous task uses the full route.

The route decision and reason are persisted. AgentShell behavior is unchanged:
bare text and `/ask` still run Protocol Mode, streaming still exposes selected
Agent outputs, and `/compare` still runs the two modes fairly.

### 4. Compact Memory Evidence

Retriever output is deduplicated by memory identity and normalized content.
Only evidence above the configured relevance floor is injected. Evidence is
ordered by score and packed into a bounded digest containing source identity,
task topic, concise fact, and score.

Memory retrieval telemetry distinguishes candidates, accepted evidence,
deduplicated evidence, and injected bytes. Empty or low-confidence retrieval
does not consume an LLM context budget.

### 5. Prompt Contracts

Prompts become concise role contracts:

- Planner returns route and intent without restating the task.
- Retriever returns evidence facts rather than prose commentary.
- Executor returns executable code or structured validation observations.
- Summarizer answers the user directly and includes required task facts,
  computed outputs, or execution evidence.

Prompt versions are recorded in the manifest. Prompt changes must have
contract tests using fake LLM clients; tests must not depend on live provider
responses.

## Data Flow

1. Benchmark loads the suite and the P3 LLM experiment profile.
2. The manifest records model, prompt, route-policy, and budget fingerprints.
3. Text and Protocol modes execute in alternating paired order.
4. Protocol routing selects the minimal safe Agent chain.
5. Each stage receives its role-specific bounded context.
6. The existing P2 quality evaluator scores both final answers with the same
   task rule.
7. The benchmark writes per-stage context, routing, quality, token, and latency
   evidence.
8. A comparison report applies the quality guard before presenting efficiency
   gains.

## Error Handling

- Missing LLM configuration fails the real-LLM experiment before any task
  starts and prints only non-secret configuration diagnostics.
- Provider timeout or error marks the pair incomplete; it is not converted
  into a zero-quality completed sample.
- Baseline/candidate fingerprint mismatch blocks comparison.
- Invalid context budgets fail configuration validation.
- If compaction cannot retain required task content, Protocol Mode falls back
  to the uncompressed safe input and records the fallback.
- Dynamic-route uncertainty uses the full route.

## Testing Strategy

### Deterministic tests

- Budget packing retains required facts and trims evidence in priority order.
- Route policy skips only safe stages and falls back to the full route for
  ambiguous tasks.
- Memory evidence is relevance-filtered, deduplicated, bounded, and audited.
- Manifest and comparison validation reject incompatible experiments.
- Quality guard rejects a candidate with lower mean or pass rate.
- AgentShell command routing, streaming, code fidelity, and error recovery
  remain unchanged.

### Real LLM evidence

- Run the standard suite three times for the baseline.
- Apply the approved P3 optimization.
- Run the same standard suite three times for the candidate.
- Compare quality mean/pass rate first, then Agent I/O tokens and latency
  P50/P95.
- Preserve both artifact sets and record exact measured results in
  `docs/progress.md`.

## Scope Boundaries

P3 does not:

- optimize deterministic fallback answer quality;
- change providers or models between baseline and candidate;
- introduce parallel Agent execution;
- redesign AgentShell commands or interaction flow;
- claim cost savings from estimated tokens when provider usage is unavailable;
- tune long-context or showcase suites before the standard-suite quality gate
  passes.

## Deliverables

- Versioned real-LLM experiment profile and comparable artifact identities.
- Role-specific context budget implementation and telemetry.
- Quality-safe route policy with persisted reasons.
- Compact, filtered memory evidence injection.
- Concise prompt contracts and prompt-version provenance.
- Baseline-versus-candidate quality-guarded comparison report.
- Full automated regression coverage and three-repeat real-LLM evidence.
