# AgentMesh-Runtime 项目落地计划书

> 项目名称：AgentMesh-Runtime  
> 项目定位：面向多智能体协作的协议化状态传递与共享记忆运行时  
> 目标系统：openEuler 24.03-LTS-SP3  
> 包管理工具：uv  
> 推荐语言：Python 3.11+  
> 交付形态：可运行 CLI 原型 + 协议日志 + StateRef 状态流 + 共享记忆 + 双模式 Benchmark + 实验报告

---

## 1. 项目目标

AgentMesh-Runtime 的目标不是实现一个普通的多 Agent 应用，而是实现一套面向 Agent 间协作的轻量级运行时基础设施。

系统核心思想是：

```text
从 Language-Passing 转向 State-Passing
```

也就是让 Agent 之间不再反复传递大段自然语言上下文，而是通过结构化协议传递：

- 动作类型；
- 输入参数；
- 返回结果；
- 能力描述；
- 状态引用；
- 记忆引用；
- trace_id；
- error 信息。

项目最终需要证明：

1. Agent 间可以通过结构化协议完成协作；
2. 系统支持握手、能力发现、协议映射；
3. 系统支持 Text Mode 和 Protocol Mode 两种协作模式；
4. Protocol Mode 不靠长文本透传，而是通过 StateRef / EmbeddingRef / MemoryRef / BlobRef 传递中间状态；
5. 系统实现 Shared Memory Store，支持关键词、标签、语义检索；
6. 系统能在相同任务条件下进行可复现 Benchmark；
7. 最终代码能在 openEuler 24.03-LTS-SP3 上安装、编译、运行和测试。

---

## 2. 赛题要求与项目实现映射

| 赛题要求 | 本项目实现 |
|---|---|
| 面向 Agent 间协作的结构化通信机制 | 实现 Agent Message Protocol，简称 AMP |
| 通信内容包含动作类型、输入参数、返回结果和能力描述 | AMPMessage 包含 action、params、result、capability |
| 支持握手、能力发现或协议映射 | 实现 HELLO、CAPABILITY_ADVERTISE、CAPABILITY_QUERY、PROTOCOL_MAP |
| 不得仅通过自然语言长文本透传协作信息 | Protocol Mode 只传结构化消息和 StateRef |
| 支持纯文本协作模式和结构化协议协作模式 | 实现 Text Mode 与 Protocol Mode |
| 在相同任务条件下完成可复现实验对比 | Benchmark Runner 固定任务、Agent、初始记忆和工具环境 |
| 实现非文本中间状态传递 | 实现 EmbeddingState、EmbeddingRef、BlobRef、CodeResultState |
| 说明状态生成、传递、接收和使用方式 | State Exchange Layer 记录完整生命周期 |
| 实现共享记忆模块 | Shared Memory Store 保存 MemoryUnit |
| 每条记忆包含完整元数据 | memory_id、source_agent、created_at、task_topic、summary、tags、evidence_refs |
| 支持关键词、标签或语义检索 | SQLite FTS、Tag Index、HashEmbedding + cosine similarity |
| 不同 Agent 可复用历史记忆 | Planner、Retriever、Executor、Summarizer 都可访问 MemoryStore |

---

## 3. 总体架构

```text
┌──────────────────────────────────────────────┐
│ CLI / Demo / Benchmark Report                │
│ 任务运行、模式切换、实验报告、图表展示          │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ Multi-Agent Runtime                          │
│ Planner / Retriever / Executor / Summarizer  │
│ Agent 注册、调度、生命周期管理                  │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ Agent Message Protocol Layer                 │
│ HELLO / CAPABILITY / INVOKE / RESULT / ERROR │
│ 握手、能力发现、动作调用、协议映射、错误处理      │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ State Exchange Layer                         │
│ StateRef / EmbeddingRef / BlobRef             │
│ 非文本状态生成、传递、接收、复用                 │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ State Store & State Lineage                  │
│ 状态存储、状态血缘、依赖关系、生命周期管理        │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ Shared Memory Store                          │
│ SQLite FTS + Tag Index + Vector Search        │
│ 记忆存储、检索、评分、验证、复用                 │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ Sandbox & Tool Execution Layer               │
│ CodeAct Runner / Tool Executor                │
│ 代码执行、工具调用、结构化结果回传               │
└──────────────────────────────────────────────┘
                    │
┌──────────────────────────────────────────────┐
│ Evaluation & Trace Layer                     │
│ token、字符数、协议字节、耗时、命中率、质量评分   │
└──────────────────────────────────────────────┘
```

---

## 4. 技术选型

### 4.1 基础选型

| 模块 | 技术 |
|---|---|
| 语言 | Python 3.11+ |
| 包管理 | uv |
| CLI | Typer |
| 数据模型 | Pydantic v2 |
| 日志展示 | Rich |
| JSON 序列化 | orjson |
| 配置文件 | PyYAML |
| 状态 payload 序列化 | msgpack / json / raw file |
| 共享记忆数据库 | SQLite |
| 全文检索 | SQLite FTS5 |
| 语义向量 | 第一版自研 HashEmbedding |
| 代码执行 | subprocess + tempfile + timeout |
| 测试 | pytest |
| 代码检查 | ruff |
| 类型检查 | mypy |
| Dashboard | 第二阶段可选 FastAPI |

### 4.2 第一版不建议引入的组件

| 暂不引入 | 原因 |
|---|---|
| Redis | 增加部署成本，第一版 SQLite 足够 |
| Milvus / Faiss | openEuler 编译部署复杂，第一版本地余弦检索足够 |
| Kafka / RabbitMQ | 结构化通信可以先用内存事件总线和 JSONL 日志实现 |
| LangChain | 容易削弱自研运行时的创新点 |
| 大模型 API 强依赖 | Benchmark 不可复现，网络和 API Key 不稳定 |
| Docker 强依赖 | 比赛要求 openEuler 原生可运行，Docker 只能作为可选补充 |
| numpy / torch / sentence-transformers | 第一版避免重依赖，提高 openEuler 兼容性 |

---

## 5. uv 依赖配置

项目根目录使用 `pyproject.toml`：

```toml
[project]
name = "agentmesh-runtime"
version = "0.1.0"
description = "A lightweight runtime for structured multi-agent communication, state passing and shared memory."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7.0",
    "typer>=0.12.0",
    "rich>=13.7.0",
    "orjson>=3.10.0",
    "pyyaml>=6.0.1",
    "msgpack>=1.0.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]

dashboard = [
    "fastapi>=0.111.0",
    "uvicorn>=0.30.0",
]

[project.scripts]
agentmesh = "agentmesh.cli:app"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --disable-warnings"
```

---

## 6. openEuler 24.03-LTS-SP3 部署方案

### 6.1 系统依赖

`scripts/setup_openeuler.sh`：

```bash
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
```

### 6.2 标准运行命令

```bash
uv sync

uv run agentmesh --help

uv run agentmesh init

uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt

uv run agentmesh run --mode text --task examples/tasks/A1_requirements.txt

uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml

uv run agentmesh report --run runs/latest

uv run pytest
uv run ruff check .
uv run mypy src
```

---

## 7. 推荐目录结构

```text
agentmesh-runtime/
├── pyproject.toml
├── uv.lock
├── README.md
├── .agent.md
├── Makefile
├── scripts/
│   ├── setup_openeuler.sh
│   ├── run_demo.sh
│   └── run_tests.sh
├── src/
│   └── agentmesh/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── protocol/
│       │   ├── __init__.py
│       │   ├── enums.py
│       │   ├── schema.py
│       │   ├── codec.py
│       │   ├── router.py
│       │   └── capability.py
│       │
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── registry.py
│       │   ├── scheduler.py
│       │   ├── event_bus.py
│       │   └── orchestrator.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── planner.py
│       │   ├── retriever.py
│       │   ├── executor.py
│       │   └── summarizer.py
│       │
│       ├── state/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── store.py
│       │   ├── refs.py
│       │   ├── embedding.py
│       │   └── lineage.py
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── sqlite_store.py
│       │   ├── search.py
│       │   ├── scorer.py
│       │   └── policy.py
│       │
│       ├── modes/
│       │   ├── __init__.py
│       │   ├── text_mode.py
│       │   └── protocol_mode.py
│       │
│       ├── sandbox/
│       │   ├── __init__.py
│       │   ├── runner.py
│       │   └── limits.py
│       │
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── benchmark.py
│       │   ├── report.py
│       │   └── quality.py
│       │
│       └── storage/
│           ├── __init__.py
│           ├── paths.py
│           └── jsonl.py
│
├── examples/
│   ├── tasks/
│   │   ├── A1_requirements.txt
│   │   ├── A2_architecture.txt
│   │   ├── A3_modules.txt
│   │   ├── A4_readme.txt
│   │   ├── A5_report.txt
│   │   ├── B1_repo_analysis.txt
│   │   ├── B2_tests.txt
│   │   ├── B3_bugfix.txt
│   │   ├── B4_api_docs.txt
│   │   └── B5_perf_report.txt
│   ├── corpora/
│   │   ├── contest_requirements.md
│   │   └── sample_repo_summary.md
│   └── benchmarks/
│       └── continuous_tasks.yaml
│
├── tests/
│   ├── test_protocol_schema.py
│   ├── test_capability_handshake.py
│   ├── test_state_store.py
│   ├── test_embedding_ref.py
│   ├── test_memory_store.py
│   ├── test_sandbox_runner.py
│   ├── test_text_mode.py
│   ├── test_protocol_mode.py
│   └── test_benchmark.py
│
└── docs/
    ├── architecture.md
    ├── protocol_spec.md
    ├── state_lifecycle.md
    ├── memory_design.md
    ├── benchmark_design.md
    └── openeuler_deploy.md
```

---

## 8. 核心模块设计

## 8.1 Agent Message Protocol

AMP 是整个项目的核心。

### 消息类型

```python
from enum import StrEnum

class MsgType(StrEnum):
    HELLO = "HELLO"
    CAPABILITY_ADVERTISE = "CAPABILITY_ADVERTISE"
    CAPABILITY_QUERY = "CAPABILITY_QUERY"
    PROTOCOL_MAP = "PROTOCOL_MAP"
    INVOKE = "INVOKE"
    RESULT = "RESULT"
    ERROR = "ERROR"
    MEMORY_PUT = "MEMORY_PUT"
    MEMORY_QUERY = "MEMORY_QUERY"
    STATE_REF = "STATE_REF"
    TRACE_EVENT = "TRACE_EVENT"
```

### 消息结构

```python
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

class AMPMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg-{uuid4().hex[:12]}")
    trace_id: str
    source_agent: str
    target_agent: str
    msg_type: str
    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    capability: list[str] = Field(default_factory=list)
    state_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### 必须支持的协议动作

```text
system.hello
system.capability_advertise
system.capability_query
protocol.map
memory.put
memory.keyword_search
memory.tag_search
memory.semantic_search
state.put_text
state.put_embedding
state.get
tool.run_python
trace.record
```

---

## 8.2 Agent 角色

系统默认启用四个 Agent：

| Agent | 职责 | 关键能力 |
|---|---|---|
| PlannerAgent | 任务理解、拆解、查询历史记忆、生成计划 | plan.create、memory.query、protocol.map |
| RetrieverAgent | 文档检索、记忆检索、证据收集 | memory.keyword_search、memory.tag_search、memory.semantic_search |
| ExecutorAgent | 工具调用、代码执行、结果验证 | tool.run_python、tool.validate_result |
| SummarizerAgent | 汇总结果、生成最终答案、写入记忆 | summary.create、memory.put |

### BaseAgent

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    name: str
    capabilities: list[str]

    def hello(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "capabilities": self.capabilities,
        }

    @abstractmethod
    def handle(self, message, context):
        raise NotImplementedError
```

第一版不做复杂并发，使用同步链路即可：

```text
PlannerAgent
  ↓
RetrieverAgent
  ↓
ExecutorAgent
  ↓
SummarizerAgent
```

---

## 8.3 StateStore 与 StateRef

### 状态类型

```python
from enum import StrEnum

class StateType(StrEnum):
    TEXT = "text"
    EMBEDDING = "embedding"
    SUMMARY = "summary"
    EVIDENCE = "evidence"
    CODE_RESULT = "code_result"
    BLOB = "blob"
    HIDDEN = "hidden"
```

### StateRecord

```python
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

class StateRecord(BaseModel):
    state_id: str = Field(default_factory=lambda: f"state-{uuid4().hex[:12]}")
    trace_id: str
    state_type: str
    producer: str
    consumers: list[str] = Field(default_factory=list)
    parent_state_refs: list[str] = Field(default_factory=list)
    payload_ref: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def ref(self) -> str:
        return f"state://{self.state_type}/{self.state_id}"
```

### State 生命周期

```text
Agent 生成中间结果
  ↓
StateStore 写入 payload
  ↓
StateStore 生成 StateRecord
  ↓
返回 state://type/id
  ↓
AMPMessage 只传 state_refs
  ↓
下游 Agent 根据 StateRef 读取真实状态
  ↓
StateLineage 记录依赖、生产者、消费者
```

---

## 8.4 HashEmbedding

第一版用本地 HashEmbedding，不调用外部模型。

### 生成流程

```text
文本
  ↓
分词
  ↓
sha256(token)
  ↓
映射到 384 维桶
  ↓
根据哈希奇偶决定 +1 / -1
  ↓
L2 normalize
  ↓
写入 StateStore
  ↓
生成 EmbeddingRef
```

### 设计理由

1. 不依赖网络；
2. 不依赖 GPU；
3. 不需要下载模型；
4. 适合 openEuler 原生部署；
5. Benchmark 结果可复现；
6. 足以证明“非文本中间表示传递”。

---

## 8.5 Shared Memory Store

### MemoryUnit

```python
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

class MemoryUnit(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem-{uuid4().hex[:12]}")
    source_agent: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None
    task_topic: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    state_refs: list[str] = Field(default_factory=list)
    embedding_ref: str | None = None
    reuse_count: int = 0
    confidence: float = 0.0
    validity_score: float = 0.0
    reuse_policy: str = "verify"
    provenance_trace_id: str
```

### SQLite 表

```sql
CREATE TABLE IF NOT EXISTS memory_units (
    memory_id TEXT PRIMARY KEY,
    source_agent TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    task_topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    state_refs_json TEXT NOT NULL,
    embedding_ref TEXT,
    reuse_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0,
    validity_score REAL DEFAULT 0,
    reuse_policy TEXT DEFAULT 'verify',
    provenance_trace_id TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
USING fts5(memory_id, task_topic, summary, tags);
```

### 检索方式

| 检索方式 | 实现 |
|---|---|
| 关键词检索 | SQLite FTS5 |
| 标签检索 | tags_json 匹配，后续可独立 tag 表优化 |
| 语义检索 | HashEmbedding + cosine similarity |

### Memory Score

```text
MemoryScore =
  0.45 × SemanticSimilarity
+ 0.20 × TagMatchScore
+ 0.15 × Confidence
+ 0.15 × EvidenceCoverage
- 0.05 × TimeDecay
```

---

## 8.6 Sandbox Runner

ExecutorAgent 的代码执行结果必须写成 CodeResultState，不能直接通过长文本传递。

### 执行流程

```text
ExecutorAgent 接收 tool.run_python
  ↓
SandboxRunner 创建临时目录
  ↓
写入 main.py
  ↓
subprocess.run 执行
  ↓
限制 timeout、cwd、stdout、stderr 长度
  ↓
捕获 stdout、stderr、exit_code、latency_ms
  ↓
写入 CodeResultState
  ↓
返回 execution_result_ref
```

---

## 9. Text Mode 与 Protocol Mode

## 9.1 Text Mode

Text Mode 是 baseline，保留长文本传递：

```text
Planner 接收完整任务
Retriever 接收完整任务 + 完整计划
Executor 接收完整任务 + 完整计划 + 完整证据
Summarizer 接收完整任务 + 完整计划 + 完整证据 + 完整执行结果
```

记录指标：

```text
message_count
text_chars
estimated_tokens
latency_ms
answer_quality_score
```

---

## 9.2 Protocol Mode

Protocol Mode 使用结构化协议和状态引用：

```text
任务文本写入 TextState
  ↓
生成 Query EmbeddingState
  ↓
Planner 发送 INVOKE 给 Retriever
  ↓
Retriever 根据 query_ref 检索记忆和语料
  ↓
Retriever 写 EvidenceState
  ↓
Executor 读取 state_refs 并执行工具
  ↓
Executor 写 CodeResultState
  ↓
Summarizer 汇总结果
  ↓
Summarizer 写 MemoryUnit
```

记录指标：

```text
message_count
protocol_bytes
state_transfer_count
state_transfer_bytes
memory_query_count
memory_hit_count
memory_hit_rate
latency_ms
answer_quality_score
```

---

## 10. Benchmark 设计

### 10.1 Benchmark 配置

`examples/benchmarks/continuous_tasks.yaml`：

```yaml
name: continuous_tasks
repeat: 3

tasks:
  - id: A1
    group: A
    topic: "contest requirement analysis"
    input_file: "examples/tasks/A1_requirements.txt"
    tags: ["requirements", "multi-agent", "protocol"]

  - id: A2
    group: A
    topic: "runtime architecture design"
    input_file: "examples/tasks/A2_architecture.txt"
    tags: ["architecture", "state-transfer", "memory"]

  - id: A3
    group: A
    topic: "module decomposition"
    input_file: "examples/tasks/A3_modules.txt"
    tags: ["modules", "runtime"]

  - id: A4
    group: A
    topic: "readme generation"
    input_file: "examples/tasks/A4_readme.txt"
    tags: ["readme", "documentation"]

  - id: A5
    group: A
    topic: "experiment report draft"
    input_file: "examples/tasks/A5_report.txt"
    tags: ["benchmark", "report"]

  - id: B1
    group: B
    topic: "python repository analysis"
    input_file: "examples/tasks/B1_repo_analysis.txt"
    tags: ["code", "repo-analysis"]

  - id: B2
    group: B
    topic: "unit test generation"
    input_file: "examples/tasks/B2_tests.txt"
    tags: ["test", "code"]

  - id: B3
    group: B
    topic: "bug fixing"
    input_file: "examples/tasks/B3_bugfix.txt"
    tags: ["bugfix", "code"]

  - id: B4
    group: B
    topic: "api documentation"
    input_file: "examples/tasks/B4_api_docs.txt"
    tags: ["api", "documentation"]

  - id: B5
    group: B
    topic: "performance report"
    input_file: "examples/tasks/B5_perf_report.txt"
    tags: ["performance", "report"]
```

### 10.2 输出文件

```text
runs/latest/
├── benchmark_summary.csv
├── benchmark_detail.jsonl
├── experiment_report.md
├── text/
│   ├── messages.jsonl
│   └── trace.jsonl
└── protocol/
    ├── messages.jsonl
    ├── states.jsonl
    ├── memory.jsonl
    └── trace.jsonl
```

### 10.3 核心指标

```text
TokenSavingRate =
(TextModeTokens - ProtocolModeTokens) / TextModeTokens × 100%

LatencyReductionRate =
(TextModeLatency - ProtocolModeLatency) / TextModeLatency × 100%

MemoryHitRate =
MemoryHitCount / MemoryQueryCount × 100%

QualityPreservationRate =
ProtocolModeQualityScore / TextModeQualityScore × 100%
```

---

## 11. CLI 设计

```bash
agentmesh init

agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt

agentmesh run --mode text --task examples/tasks/A1_requirements.txt

agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml

agentmesh memory search --keyword protocol

agentmesh memory search --tag shared-memory

agentmesh memory search --semantic "agent state passing"

agentmesh trace show --run runs/latest

agentmesh report --run runs/latest
```

---

## 12. Makefile

```makefile
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
```

---

## 13. Codex 开发阶段计划

## 阶段 0：仓库初始化

目标：

```text
uv sync 能成功
agentmesh --help 能运行
pytest 空测试能通过
```

任务：

```text
创建 pyproject.toml、src/agentmesh、tests、README.md、Makefile。
实现 Typer CLI。
添加 init、run、benchmark、report 子命令空壳。
```

验收：

```bash
uv sync
uv run agentmesh --help
uv run pytest
```

---

## 阶段 1：AMP 协议

目标：

```text
AMPMessage 可创建、校验、序列化、反序列化。
```

任务：

```text
实现 protocol/enums.py
实现 protocol/schema.py
实现 protocol/codec.py
实现 tests/test_protocol_schema.py
```

验收：

```bash
uv run pytest tests/test_protocol_schema.py
```

---

## 阶段 2：Agent 注册与能力发现

目标：

```text
四个 Agent 能注册，并输出 HELLO 和 CAPABILITY_ADVERTISE。
```

任务：

```text
实现 BaseAgent
实现 AgentRegistry
实现 CapabilityRegistry
实现 PlannerAgent、RetrieverAgent、ExecutorAgent、SummarizerAgent 最小版本
```

验收：

```bash
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
```

必须看到：

```text
HELLO
CAPABILITY_ADVERTISE
```

---

## 阶段 3：StateStore / StateRef

目标：

```text
TextState、EmbeddingState、EvidenceState、CodeResultState 能写入并返回 state:// 引用。
```

任务：

```text
实现 StateRecord
实现 StateStore
实现 StateRef parser
实现 payload 落盘
实现 state_index.sqlite
实现 lineage 记录
```

验收：

```bash
uv run pytest tests/test_state_store.py
```

---

## 阶段 4：HashEmbedding

目标：

```text
实现可复现 embedding，并支持 cosine similarity。
```

任务：

```text
实现 HashEmbeddingEncoder
实现 cosine_similarity
实现 embedding 写入 StateStore
```

验收：

```bash
uv run pytest tests/test_embedding_ref.py
```

---

## 阶段 5：Shared Memory Store

目标：

```text
记忆能写入、关键词检索、标签检索、语义检索。
```

任务：

```text
实现 MemoryUnit
实现 SQLiteMemoryStore
实现 memory_fts
实现 MemoryScorer
实现 MemoryWritePolicy
```

验收：

```bash
uv run pytest tests/test_memory_store.py
```

---

## 阶段 6：Protocol Mode

目标：

```text
跑通结构化协作完整链路。
```

任务：

```text
任务写入 TextState
Planner 生成计划
Retriever 检索记忆和证据
Executor 执行工具
Summarizer 写 MemoryUnit
全链路写 JSONL trace
```

验收：

```bash
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
```

---

## 阶段 7：Text Mode

目标：

```text
跑通纯文本协作 baseline。
```

任务：

```text
实现完整文本上下文在 Agent 间传递
统计 text_chars 和 estimated_tokens
写入 text/messages.jsonl
```

验收：

```bash
uv run agentmesh run --mode text --task examples/tasks/A1_requirements.txt
```

---

## 阶段 8：Benchmark Runner

目标：

```text
同一任务条件下对比 Text Mode 和 Protocol Mode。
```

任务：

```text
读取 YAML suite
重复执行任务
统计指标
输出 benchmark_summary.csv 和 benchmark_detail.jsonl
```

验收：

```bash
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
```

---

## 阶段 9：Report Generator

目标：

```text
自动生成 experiment_report.md。
```

任务：

```text
读取 benchmark_summary.csv
生成对比表
生成指标说明
嵌入协议日志样例
嵌入 StateRef 样例
嵌入 MemoryUnit 样例
```

验收：

```bash
uv run agentmesh report --run runs/latest
```

---

## 阶段 10：openEuler 适配

目标：

```text
在 openEuler 24.03-LTS-SP3 上完整运行。
```

任务：

```text
补充 setup_openeuler.sh
补充 docs/openeuler_deploy.md
补充 run_demo.sh
补充 run_tests.sh
保证 make all 通过
```

验收：

```bash
make all
```

---

## 14. 最终交付清单

```text
源码：
- src/agentmesh/

依赖：
- pyproject.toml
- uv.lock

运行脚本：
- scripts/setup_openeuler.sh
- scripts/run_demo.sh
- scripts/run_tests.sh
- Makefile

文档：
- README.md
- .agent.md
- docs/architecture.md
- docs/protocol_spec.md
- docs/state_lifecycle.md
- docs/memory_design.md
- docs/benchmark_design.md
- docs/openeuler_deploy.md

样例：
- examples/tasks/
- examples/corpora/
- examples/benchmarks/continuous_tasks.yaml

测试：
- tests/test_protocol_schema.py
- tests/test_capability_handshake.py
- tests/test_state_store.py
- tests/test_embedding_ref.py
- tests/test_memory_store.py
- tests/test_sandbox_runner.py
- tests/test_text_mode.py
- tests/test_protocol_mode.py
- tests/test_benchmark.py

实验输出：
- runs/sample/benchmark_summary.csv
- runs/sample/benchmark_detail.jsonl
- runs/sample/experiment_report.md
```

---

## 15. 答辩展示重点

比赛答辩时建议展示四个画面：

### 15.1 协议通信日志

展示：

```text
HELLO
CAPABILITY_ADVERTISE
INVOKE
RESULT
MEMORY_PUT
STATE_REF
```

证明系统不是自然语言透传。

### 15.2 StateRef 状态流

展示：

```text
state://text/...
state://embedding/...
state://evidence/...
state://code_result/...
```

证明非文本中间状态可以跨 Agent 传递。

### 15.3 Shared Memory 检索

展示：

```bash
agentmesh memory search --keyword protocol
agentmesh memory search --tag shared-memory
agentmesh memory search --semantic "state passing"
```

证明系统支持关键词、标签、语义检索。

### 15.4 Benchmark 对比图表

展示：

```text
TokenSavingRate
LatencyReductionRate
MemoryHitRate
QualityPreservationRate
```

证明 Protocol Mode 相比 Text Mode 具备可量化优势。

---

## 16. 参考资料

- openEuler 24.03-LTS-SP3 官方下载与版本说明：https://www.openeuler.org/en/download/archive/detail/?version=openEuler+24.03+LTS+SP3
- uv Locking and Syncing：https://docs.astral.sh/uv/concepts/projects/sync/
- uv Working on Projects：https://docs.astral.sh/uv/guides/projects/
- SQLite FTS5：https://sqlite.org/fts5.html
