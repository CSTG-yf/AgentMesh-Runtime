# AgentMesh-Runtime Linux Release

This release package is designed for Linux and openEuler-style reproduction.
It runs the AgentMesh Python runtime with `uv` and starts the TEI embedding
model through Docker.

## Quick Start

```bash
tar -xzf agentmesh-runtime-linux-x86_64.tar.gz
cd agentmesh-runtime-linux-x86_64
./install.sh
./start-shell.sh
```

`install.sh` performs these actions:

- creates `.env` from `.env.example`;
- sets `AGENTMESH_EMBEDDING_PROVIDER=tei`;
- installs Python dependencies from `uv.lock`;
- starts `docker-compose.embedding.yml`;
- initializes `runs/latest`;
- verifies `uv run agentmesh --help`.

Important: the release package never includes a real LLM API key. `install.sh`
only creates a template `.env`. To use a real model, edit `.env` after
installation:

```bash
AGENTMESH_LLM_BASE_URL=https://api.example.com/v1
AGENTMESH_LLM_API_KEY=sk-...
AGENTMESH_LLM_MODEL=your-model
```

For offline reproducible runs, leave these values empty and use `--no-llm`.

## Common Commands

```bash
./start-shell.sh
./run-benchmark.sh
./stop-services.sh
```

Inside the shell:

```text
/help
/ask 解释 AgentMesh 的 StateRef 机制
/compare --no-llm 生成一个多步骤评测方案
/benchmark --no-llm standard
/dashboard
/exit
```

## Requirements

- Linux x86_64
- Python 3.11+
- Docker with the Compose plugin
- Network access for the first TEI image/model pull
- `curl` if `uv` is not already installed

## Benchmark Artifacts

If the source workspace contains generated benchmark results, the release
builder copies them into:

```text
benchmark-results/
```

Fresh runs write artifacts to:

```text
runs/latest/benchmarks/
```

## LLM Settings

The release works offline with `--no-llm`. To enable a real model, edit `.env`:

```bash
AGENTMESH_LLM_BASE_URL=https://api.example.com/v1
AGENTMESH_LLM_API_KEY=sk-...
AGENTMESH_LLM_MODEL=your-model
```
