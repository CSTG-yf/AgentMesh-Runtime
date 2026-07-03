# P3 LLM Results

## Baseline

- Status: BLOCKED
- Profile: `p3-baseline-v1`
- Experiment ID: `exp-b1cb546b9a71e580`
- Manifest SHA-256: `73698776c5dff062d25d7dd752c79786863c916110348e3006aaadb6c47ed59c`
- Model fingerprint: `17cbdae78432071d`
- Profile SHA-256: `47232830034711c7dab94348cee4c1ec6da4c1237886242705b10ea99d5d089c`
- Suite SHA-256: `1f2e5c97547f8cbe50e763d90615b5591e19184d9c50d1b03643cd19369fe3cb`
- Quality rules SHA-256: `6b300c3fab56502bb52bc2940b69038e87966fc43f3d6f52d0296addea847f5d`
- Manifest status: `failed`
- Completed samples: 8 / 18 paired logical samples / 16 detail rows
- Failure short code: `llm_provider_error`
- Blocker: the real LLM provider failed during round 2 task `SA3`; the integrity
  gate aborted the retry and did not append the failed pair.
- Protocol quality mean/pass: unavailable (incomplete baseline)
- Protocol token mean: unavailable (incomplete baseline)
- Protocol latency P50/P95: unavailable (incomplete baseline)
