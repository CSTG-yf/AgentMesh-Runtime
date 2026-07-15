#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
PACKAGE_NAME="${1:-agentmesh-runtime-linux-x86_64}"
PACKAGE_DIR="$DIST_DIR/$PACKAGE_NAME"
ARCHIVE_PATH="$DIST_DIR/$PACKAGE_NAME.tar.gz"

cd "$ROOT_DIR"

rm -rf "$PACKAGE_DIR" "$ARCHIVE_PATH"
mkdir -p "$PACKAGE_DIR" "$DIST_DIR"

tar \
  --exclude='./.git' \
  --exclude='./.agents' \
  --exclude='./.codex' \
  --exclude='./.venv' \
  --exclude='./.mypy_cache' \
  --exclude='./.pytest_cache' \
  --exclude='./.ruff_cache' \
  --exclude='./dist' \
  --exclude='./build' \
  --exclude='./target' \
  --exclude='./data' \
  --exclude='./runs' \
  --exclude='./.env' \
  --exclude='./.env.*' \
  --exclude='./generated_code.py' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -cf - . | tar -x -C "$PACKAGE_DIR"

for file in \
  .dockerignore \
  .env.example \
  .gitignore \
  .python-version \
  Cargo.lock \
  Cargo.toml \
  docker-compose.embedding.yml \
  docker-compose.yml \
  Dockerfile \
  Makefile \
  pyproject.toml \
  README.md \
  uv.lock
do
  if [ -f "$ROOT_DIR/$file" ]; then
    cp "$ROOT_DIR/$file" "$PACKAGE_DIR/$file"
  fi
done

cp release/install.sh "$PACKAGE_DIR/install.sh"
cp release/start-shell.sh "$PACKAGE_DIR/start-shell.sh"
cp release/run-benchmark.sh "$PACKAGE_DIR/run-benchmark.sh"
cp release/stop-services.sh "$PACKAGE_DIR/stop-services.sh"
cp release/README_RELEASE.md "$PACKAGE_DIR/README_RELEASE.md"

chmod +x \
  "$PACKAGE_DIR/install.sh" \
  "$PACKAGE_DIR/start-shell.sh" \
  "$PACKAGE_DIR/run-benchmark.sh" \
  "$PACKAGE_DIR/stop-services.sh"

if [ -d "$ROOT_DIR/runs/latest/benchmarks" ]; then
  mkdir -p "$PACKAGE_DIR/benchmark-results"
  cp -R "$ROOT_DIR/runs/latest/benchmarks/." "$PACKAGE_DIR/benchmark-results/"
fi

tar -C "$DIST_DIR" -czf "$ARCHIVE_PATH" "$PACKAGE_NAME"

printf 'Release archive written:\n  %s\n' "$ARCHIVE_PATH"
