# Benchmark Design

The benchmark runner executes the same tasks in Text Mode and Protocol Mode.

Tracked metrics:

- message count
- text characters and estimated tokens
- protocol bytes
- state transfer count and bytes
- memory query count and hit count
- latency
- deterministic quality score

Generated artifacts:

- `runs/latest/benchmark_summary.csv`
- `runs/latest/benchmark_detail.jsonl`
- `runs/latest/experiment_report.md`
