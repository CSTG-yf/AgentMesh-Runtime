FROM openeuler/openeuler:24.03-lts-sp3

WORKDIR /app

ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN dnf update -y && \
    dnf install -y \
      cargo \
      curl \
      gcc \
      gcc-c++ \
      git \
      make \
      python3 \
      python3-devel \
      rust \
      sqlite \
      sqlite-devel && \
    dnf clean all && \
    rm -rf /var/cache/dnf

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml uv.lock README.md Cargo.toml Cargo.lock ./
COPY src ./src
COPY crates ./crates
COPY prompts ./prompts
COPY examples ./examples

RUN uv sync --all-extras --frozen && \
    uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml --release && \
    mkdir -p data runs/latest

ENTRYPOINT ["agentmesh"]
CMD ["--help"]
