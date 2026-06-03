.PHONY: sync test lint typecheck demo benchmark report all

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

all: sync lint typecheck test benchmark report
