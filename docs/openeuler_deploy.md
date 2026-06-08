# openEuler 24.03-LTS-SP3 Deployment

## Docker quick path

The fastest reproducible deployment is Docker Compose:

```bash
./scripts/deploy_openeuler_docker.sh
docker compose run --rm agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
docker compose run --rm agentmesh report --run runs/latest
```

This builds the AgentMesh CLI in an openEuler-based image and starts TEI
embedding as a sidecar service. See [docker_deploy.md](docker_deploy.md).

## Bare-metal path

Install system dependencies:

```bash
sudo dnf update -y
sudo dnf install -y python3 python3-devel gcc gcc-c++ make sqlite sqlite-devel git curl
```

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Run project checks:

```bash
uv sync --all-extras
uv run pytest
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
uv run agentmesh report --run runs/latest
```

Optional LLM settings can be placed in `.env` at the project root. The first MVP does not require them:

```bash
cp .env.example .env
```
