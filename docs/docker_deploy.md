# Docker Deployment

This deployment path is intended for fast openEuler reproduction. It builds the
AgentMesh CLI inside an openEuler userspace image and runs Hugging Face TEI as a
sidecar embedding service.

## Services

- `agentmesh`: openEuler-based Python CLI image with the optional Rust core built in.
- `tei-embedding`: Hugging Face Text Embeddings Inference for `BAAI/bge-small-zh-v1.5`.
- `hf-cache`: persistent model cache volume. Restarting containers will not delete the model.

Runtime artifacts are mounted from the host:

- `./data` for global memory and persistent SQLite files.
- `./runs` for benchmark outputs and reports.
- `./prompts` for editable agent prompt templates.

## One-Command Setup

On an openEuler host with Docker and the Compose plugin installed:

```bash
./scripts/deploy_openeuler_docker.sh
```

The script creates `.env` from `.env.example` if needed, builds the main image,
pulls the TEI image, and starts the embedding service.

## Common Commands

Run the interactive shell:

```bash
docker compose run --rm agentmesh shell
```

Compare Text Mode and Protocol Mode on the same prompt:

```bash
docker compose run --rm agentmesh compare "写出一个快速排序算法并输出结果数组"
```

Run the reproducible benchmark:

```bash
docker compose run --rm agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
docker compose run --rm agentmesh report --run runs/latest
```

Run with configured LLM access:

```bash
docker compose run --rm agentmesh compare "生成一个多 Agent 系统测试方案" --llm
```

LLM settings can be edited in `.env`:

```bash
AGENTMESH_LLM_BASE_URL=https://api.example.com/v1
AGENTMESH_LLM_API_KEY=sk-...
AGENTMESH_LLM_MODEL=your-model
```

## Embedding Cache

TEI downloads `BAAI/bge-small-zh-v1.5` into the named Docker volume `hf-cache`.
The cache survives container restarts and normal `docker compose down`.

To remove the cached model intentionally:

```bash
docker compose down -v
```

## Notes

Text Mode remains the plain natural-language baseline. It does not use TEI,
StateRef, memory, Rust core, or structured transport. Protocol Mode uses the
embedding service through `AGENTMESH_EMBEDDING_PROVIDER=tei` and falls back to
local hash embeddings if TEI is unavailable.
