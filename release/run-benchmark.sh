#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  printf '[agentmesh-release] .venv not found. Run ./install.sh first.\n' >&2
  exit 1
fi

docker compose -f docker-compose.embedding.yml up -d

uv run agentmesh benchmark --suite standard --no-llm
uv run agentmesh benchmark --suite long --no-llm
uv run agentmesh benchmark --suite showcase --no-llm
uv run agentmesh dashboard

printf '\nBenchmark dashboard:\n  runs/latest/benchmarks/benchmark_dashboard.html\n'
