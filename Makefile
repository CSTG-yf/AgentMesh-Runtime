.PHONY: sync test lint typecheck demo benchmark report docker-build docker-up docker-shell docker-benchmark all

sync:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

demo:
	uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt

benchmark:
	uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml

report:
	uv run agentmesh report --run runs/latest

docker-build:
	docker compose build agentmesh

docker-up:
	docker compose up -d tei-embedding

docker-shell:
	docker compose run --rm agentmesh shell

docker-benchmark:
	docker compose run --rm agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml

all: sync lint typecheck test benchmark report
