#!/usr/bin/env bash
set -euo pipefail

sudo dnf update -y
sudo dnf install -y \
  python3 \
  python3-devel \
  gcc \
  gcc-c++ \
  make \
  sqlite \
  sqlite-devel \
  git \
  curl

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

python3 --version
sqlite3 --version
uv --version

uv sync --all-extras
uv run pytest
