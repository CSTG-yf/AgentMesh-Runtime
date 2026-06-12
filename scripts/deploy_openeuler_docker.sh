#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required. On openEuler, install Docker or Podman with compose support first." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is required." >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
fi

mkdir -p data runs

docker compose pull tei-embedding
docker compose build agentmesh
docker compose up -d tei-embedding

echo "AgentMesh Docker deployment is ready."
echo
echo "Try:"
echo "  docker compose run --rm agentmesh compare \"写出一个 Python 验证脚本并输出结果\""
echo "  docker compose run --rm agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml"
echo "  docker compose run --rm agentmesh report --run runs/latest"
echo "  docker compose run --rm agentmesh shell"
