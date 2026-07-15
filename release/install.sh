#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

info() {
  printf '[agentmesh-release] %s\n' "$1"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[agentmesh-release] missing required command: %s\n' "$1" >&2
    return 1
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  need_cmd curl
  info "uv not found; installing uv with the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  need_cmd uv
}

ensure_docker_compose() {
  need_cmd docker
  if docker compose version >/dev/null 2>&1; then
    return
  fi
  printf '[agentmesh-release] Docker Compose plugin is required. Install docker compose first.\n' >&2
  exit 1
}

prepare_env() {
  if [ ! -f .env ]; then
    cp .env.example .env
  fi
  # Release defaults to the Docker TEI service. The runtime still falls back to
  # HashEmbedding if TEI is unavailable.
  sed -i 's/^AGENTMESH_EMBEDDING_PROVIDER=.*/AGENTMESH_EMBEDDING_PROVIDER=tei/' .env
  sed -i 's#^AGENTMESH_EMBEDDING_BASE_URL=.*#AGENTMESH_EMBEDDING_BASE_URL=http://127.0.0.1:8080#' .env
}

info "checking runtime prerequisites"
ensure_uv
ensure_docker_compose

info "preparing .env and runtime directories"
prepare_env
mkdir -p data runs/latest

info "installing Python dependencies from uv.lock"
uv sync --all-extras --frozen

info "starting Docker TEI embedding service"
docker compose -f docker-compose.embedding.yml up -d

info "initializing AgentMesh runtime"
uv run agentmesh init

info "running smoke check"
uv run agentmesh --help >/dev/null

info "installation complete"
printf '\nNext command:\n  ./start-shell.sh\n\n'
