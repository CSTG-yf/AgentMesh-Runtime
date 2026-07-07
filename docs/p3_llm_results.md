# P3 LLM Results

## Baseline

- Status: COMPLETE
- Profile: `p3-baseline-v1`
- Experiment ID: `exp-b1cb546b9a71e580`
- Manifest SHA-256: `56c0c24a5b151d3d8313fcbc1fcf2dce66398e38b44e43ba91b44c37cf9843d8`
- Model fingerprint: `17cbdae78432071d`
- Profile SHA-256: `47232830034711c7dab94348cee4c1ec6da4c1237886242705b10ea99d5d089c`
- Suite SHA-256: `1f2e5c97547f8cbe50e763d90615b5591e19184d9c50d1b03643cd19369fe3cb`
- Quality rules SHA-256: `6b300c3fab56502bb52bc2940b69038e87966fc43f3d6f52d0296addea847f5d`
- Manifest status: `complete`
- Completed samples: 18 / 18 paired logical samples / 36 detail rows
- Quality coverage: 18 scored / 0 unscored runs
- Protocol quality mean/pass: 0.833333 / 0.833333
- Protocol token mean: 1892.333333
- Protocol latency P50/P95: 48012.0 ms / 90377.7 ms

## Candidate v1

- Status: COMPLETE
- Profile: `p3-candidate`
- Experiment ID: `exp-0f8e16fad3e799dd`
- Manifest status: `complete`
- Completed samples: 18 / 18 paired logical samples / 36 detail rows
- Protocol quality mean/pass: 0.916667 / 0.888889
- Protocol token mean: 1425.777778
- Protocol latency P50/P95: 45450.0 ms / 126432.65 ms
- Gate result versus `p3-baseline`: failed efficiency claim because Protocol latency P95 regressed by more than 10%.

## Candidate v2

- Status: COMPLETE
- Profile: `p3-candidate-v2`
- Experiment ID: `exp-33afcdaffaf56129`
- Manifest status: `complete`
- Completed samples: 18 / 18 paired logical samples / 36 detail rows
- Model fingerprint: `8698766788a176e6`
- Profile SHA-256: `f165b6d572f78fc9f3bfe34a0d14d96315f17e84bc70b199fc5efc0a5ca20054`
- Protocol quality mean/pass: 0.842593 / 0.777778
- Text quality mean/pass: 0.777778 / 0.777778
- Quality score delta: +0.064815
- Protocol token mean: 1322.777778
- Text token mean: 1596.944444
- Token saving rate: 17.168203%
- Protocol latency P50/P95: 118515.0 ms / 169257.45 ms
- Text latency P50/P95: 138528.5 ms / 298813.3 ms
- Latency reduction rate inside the paired v2 run: 22.789655%
- Memory hit rate: 1.0
- Memory reused unit count / evidence count: 18 / 18

`benchmark-compare` rejected the old-baseline comparison because the model fingerprint changed from `17cbdae78432071d` to `8698766788a176e6`. Therefore v2 is a complete candidate run, but not a formal baseline-vs-candidate gate pass. A formal gate requires rerunning `p3-baseline` under the current model fingerprint and then rerunning:

```powershell
uv run agentmesh benchmark-compare --suite continuous_tasks --baseline p3-baseline --candidate p3-candidate-v2
```

## Current interpretation

- Code/runtime objective: achieved for P3 candidate-v2.
- Stability objective: improved. Recovered LLM errors are now retained in result metrics instead of aborting an otherwise completed pair.
- Within-run efficiency: positive. Candidate-v2 Protocol mode uses fewer tokens and lower latency than Text mode in the same run.
- Formal gate status: pending same-model baseline rerun.
