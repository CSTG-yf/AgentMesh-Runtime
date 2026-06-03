#!/usr/bin/env bash
set -euo pipefail

uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
