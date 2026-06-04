# AgentMesh-Runtime

AgentMesh-Runtime 是一个面向多 Agent 协作的轻量级运行时原型。项目重点不是做聊天机器人，也不是套用现成多 Agent 框架，而是验证一套从 `Language-Passing` 转向 `State-Passing` 的系统机制：Agent 之间通过结构化 AMP 协议、`StateRef` 状态引用、HashEmbedding、共享记忆和可复现实验评测完成协作。

本仓库第一版保持离线、确定性、轻依赖，默认不调用外部大模型 API；后续如需接入大模型，请在本地 `.env` 中配置密钥和接口地址。

## 项目功能

- Agent Message Protocol：支持 `HELLO`、`CAPABILITY_ADVERTISE`、`CAPABILITY_QUERY`、`PROTOCOL_MAP`、`INVOKE`、`RESULT`、`ERROR`、`STATE_REF`、`MEMORY_PUT` 等消息。
- 四个内置 Agent：`PlannerAgent`、`RetrieverAgent`、`ExecutorAgent`、`SummarizerAgent`。
- 双模式协作：`Text Mode` 作为长文本 baseline，`Protocol Mode` 使用结构化消息和 `state://...` 引用。
- StateStore：支持 text、embedding、summary、evidence、code_result、blob 等状态类型，并记录 lineage。
- HashEmbedding：不依赖外部模型，使用本地 hash trick 生成确定性向量。
- Shared Memory Store：基于 SQLite + FTS5，支持关键词检索、标签检索和语义检索。
- Rust Core：可选 `agentmesh_core` 扩展加速 StateRef 解析、JSON/msgpack 协议编码、HashEmbedding、语义 top-k 检索，并提供 Rust sandbox subprocess backend。
- SandboxRunner：用受限 subprocess 执行 Python 代码；默认优先复用 warm worker，Rust Core 可用时可走 Rust backend，不可用时自动回退到 Python backend，结果写入 `CodeResultState`。
- Benchmark Runner：对同一批连续任务分别运行 Text Mode 和 Protocol Mode，输出可复现实验数据。
- Report Generator：自动生成 `runs/latest/experiment_report.md`。
- openEuler 部署：提供 openEuler 24.03-LTS-SP3 安装和验证脚本。

## 快速开始

```bash
uv sync --all-extras
uv run agentmesh --help
uv run agentmesh init
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
uv run agentmesh run --mode text --task examples/tasks/A1_requirements.txt
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
uv run agentmesh report --run runs/latest
uv run agentmesh chat --message "解释一下 AgentMesh Runtime 的 StateRef 机制"
```

## 常用命令

```bash
# 初始化运行目录
uv run agentmesh init

# 运行结构化协议模式
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt

# 运行纯文本 baseline
uv run agentmesh run --mode text --task examples/tasks/A1_requirements.txt

# 连续任务评测
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml

# 生成实验报告
uv run agentmesh report --run runs/latest

# 共享记忆检索
uv run agentmesh memory search --keyword protocol
uv run agentmesh memory search --tag protocol
uv run agentmesh memory search --semantic "agent state passing"

# 查看 trace
uv run agentmesh trace show

# 单轮真实模型交互，需先配置 .env
uv run agentmesh chat --message "帮我分析这个项目当前还缺什么"

# 多轮交互式会话，需先配置 .env
uv run agentmesh chat
```

## 质量验证

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

如需启用 Rust Core 优化，请先安装 Rust 工具链，然后运行：

```bash
uv sync --extra dev
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
```

在 Windows PowerShell 中，如果同时存在 `VIRTUAL_ENV` 和 `CONDA_PREFIX` 导致 maturin 报错，可先运行：

```powershell
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
```

未安装 Rust Core 时，项目会自动回退到纯 Python 实现。
`SandboxResult.backend` 会标记当前代码执行使用的是 `warm_python`、`rust` 还是 `python` backend。

在 Linux/openEuler 环境中也可以使用：

```bash
make all
```

当前 Windows PowerShell 环境如果没有安装 `make`，请分别运行上面的 `uv` 命令。

## 大模型配置

第一版 MVP 默认不依赖外部大模型，Benchmark 也不需要网络或 API Key。若后续 Agent 策略需要调用模型，请复制示例文件：

```bash
cp .env.example .env
```

在 `.env` 中填写：

```bash
AGENTMESH_LLM_BASE_URL=
AGENTMESH_LLM_API_KEY=
AGENTMESH_LLM_MODEL=
AGENTMESH_LLM_TIMEOUT_SECONDS=30
AGENTMESH_PROMPT_DIR=
```

安全说明：

- `.env` 和 `.env.*` 已写入 `.gitignore`，不会提交真实密钥。
- `.env.example` 会保留在仓库中，作为配置模板。
- 代码中使用 `SecretStr` 保存 API Key，JSON 序列化时会脱敏。
- 当前确定性 Agent 只读取配置状态，不会主动发起外部网络调用。
- `agentmesh chat` 和已配置 LLM 的 Agent 处理链路会调用 `.env` 中的模型接口。

## Prompt 模板配置

初始提示词模板独立存储在 `prompts/` 目录：

```text
prompts/
├── executor.md       # ExecutorAgent 初始提示词
├── interactive.md    # agentmesh chat 交互式助手初始提示词
├── planner.md        # PlannerAgent 初始提示词
├── retriever.md      # RetrieverAgent 初始提示词
└── summarizer.md     # SummarizerAgent 初始提示词
```

模板支持简单变量替换：

- `planner.md` 可使用 `{task}`
- `retriever.md` 可使用 `{query}`
- `executor.md` 可使用 `{input}`
- `summarizer.md` 可使用 `{input}`
- `interactive.md` 可使用 `{user_input}`

如需使用自定义模板目录，可以在 `.env` 中设置：

```bash
AGENTMESH_PROMPT_DIR=E:/system-compute/prompts
```

## 运行产物

运行命令后会生成 `runs/latest/`：

```text
runs/latest/
├── benchmark_summary.csv       # Benchmark 汇总指标
├── benchmark_detail.jsonl      # 每个任务、每种模式的详细指标
├── experiment_report.md        # 自动生成的实验报告
├── data/
│   ├── memory.sqlite           # SQLite 共享记忆数据库
│   ├── state_index.sqlite      # SQLite 状态索引
│   └── states/                 # State payload 文件
├── protocol/
│   ├── messages.jsonl          # Protocol Mode AMP 消息日志
│   ├── memory.jsonl            # 写入共享记忆的 MemoryUnit 日志
│   ├── states.jsonl            # StateRecord 和 lineage 日志
│   └── trace.jsonl             # Protocol Mode 指标 trace
├── sandbox/                    # SandboxRunner 临时执行目录
└── text/
    ├── messages.jsonl          # Text Mode 消息日志
    └── trace.jsonl             # Text Mode 指标 trace
```

`runs/` 是运行产物目录，已被 `.gitignore` 忽略。

## 目录架构与文件说明

```text
agentmesh-runtime/
├── .agent.md                                  # Codex/Agent 项目约束：目标、架构边界、技术选型、验收标准
├── .env.example                              # 大模型 endpoint/key/model 的本地配置模板，可提交
├── .gitignore                                # 忽略 Python 缓存、虚拟环境、runs、真实 .env 等本地产物
├── .python-version                           # uv 使用的 Python 版本提示，当前为 3.11
├── AgentMesh_Runtime_项目落地计划书.md        # 项目落地计划书和功能验收来源
├── Makefile                                  # sync/test/lint/typecheck/demo/benchmark/report/all 统一命令
├── pyproject.toml                            # Python 项目元数据、依赖、CLI 入口、ruff/mypy/pytest 配置
├── README.md                                 # 项目说明、目录说明、运行方式和配置说明
├── uv.lock                                   # uv 锁文件，保证依赖可复现
├── 赛题.txt                                  # 比赛题目原文和评分要求
│
├── docs/
│   ├── architecture.md                       # 总体架构说明和模块边界
│   ├── benchmark_design.md                   # Benchmark 指标、产物和运行逻辑说明
│   ├── implementation_audit.md               # 对照计划书的功能实现审计
│   ├── memory_design.md                      # Shared Memory Store 设计说明
│   ├── openeuler_deploy.md                   # openEuler 24.03-LTS-SP3 部署说明
│   ├── progress.md                           # 阶段进度、验收命令和结果记录
│   ├── protocol_spec.md                      # AMP 协议字段和校验规则
│   └── state_lifecycle.md                    # StateRef 生成、传递、消费和 lineage 生命周期
│
├── examples/
│   ├── benchmarks/
│   │   └── continuous_tasks.yaml             # 10 个连续任务、repeat=3 的 benchmark suite
│   ├── corpora/
│   │   ├── contest_requirements.md           # 离线语料：赛题要求摘要
│   │   └── sample_repo_summary.md            # 离线语料：示例仓库摘要
│   └── tasks/
│       ├── A1_requirements.txt               # A 组任务 1：赛题需求分析
│       ├── A2_architecture.txt               # A 组任务 2：运行时架构设计
│       ├── A3_modules.txt                    # A 组任务 3：模块拆解
│       ├── A4_readme.txt                     # A 组任务 4：README 生成
│       ├── A5_report.txt                     # A 组任务 5：实验报告草稿
│       ├── B1_repo_analysis.txt              # B 组任务 1：Python 仓库分析
│       ├── B2_tests.txt                      # B 组任务 2：测试生成
│       ├── B3_bugfix.txt                     # B 组任务 3：缺陷修复流程
│       ├── B4_api_docs.txt                   # B 组任务 4：API 文档
│       └── B5_perf_report.txt                # B 组任务 5：性能报告
│
├── prompts/
│   ├── executor.md                           # ExecutorAgent 可配置初始提示词模板
│   ├── interactive.md                        # 交互式 chat 可配置初始提示词模板
│   ├── planner.md                            # PlannerAgent 可配置初始提示词模板
│   ├── retriever.md                          # RetrieverAgent 可配置初始提示词模板
│   └── summarizer.md                         # SummarizerAgent 可配置初始提示词模板
│
├── scripts/
│   ├── run_demo.sh                           # 一键运行 Protocol Mode demo
│   ├── run_tests.sh                          # 一键运行 ruff、mypy、pytest
│   └── setup_openeuler.sh                    # openEuler 系统依赖安装与基础验证脚本
│
├── src/
│   └── agentmesh/
│       ├── __init__.py                       # 包版本和包初始化
│       ├── cli.py                            # Typer CLI：init/run/benchmark/report/memory/trace
│       ├── config.py                         # `.env` 和环境变量配置读取，保存可选 LLM 参数
│       ├── errors.py                         # 统一错误类型：Protocol/State/Memory/Capability/Sandbox/Benchmark
│       │
│       ├── agents/
│       │   ├── __init__.py                   # 导出四个默认 Agent
│       │   ├── executor.py                   # ExecutorAgent：工具执行、结果验证、消费 StateRef
│       │   ├── planner.py                    # PlannerAgent：任务拆解、协议映射、记忆查询意图
│       │   ├── retriever.py                  # RetrieverAgent：记忆检索、证据收集
│       │   └── summarizer.py                 # SummarizerAgent：摘要生成、写入共享记忆
│       │
│       ├── eval/
│       │   ├── __init__.py                   # Evaluation 子包初始化
│       │   ├── benchmark.py                  # Benchmark suite 读取、双模式执行、指标汇总
│       │   ├── metrics.py                    # RunMetrics、ModeRunResult 和 token 估算
│       │   ├── quality.py                    # 确定性质量分估算
│       │   └── report.py                     # experiment_report.md 自动生成
│       │
│       ├── chat/
│       │   ├── __init__.py                   # Chat 子包初始化
│       │   └── session.py                    # 交互式 chat turn、历史消息和 prompt 组装
│       │
│       ├── llm/
│       │   ├── __init__.py                   # LLM 子包初始化
│       │   └── client.py                     # OpenAI-compatible Chat Completions 客户端
│       │
│       ├── memory/
│       │   ├── __init__.py                   # Memory 子包初始化
│       │   ├── policy.py                     # MemoryWritePolicy：控制记忆是否写入
│       │   ├── schema.py                     # MemoryUnit Pydantic 数据结构
│       │   ├── scorer.py                     # MemoryScorer：语义、标签、置信度、证据覆盖评分
│       │   ├── search.py                     # MemorySearchResult 查询结果结构
│       │   └── sqlite_store.py               # SQLiteMemoryStore：FTS5、标签检索、语义检索、复用计数
│       │
│       ├── modes/
│       │   ├── __init__.py                   # Runtime modes 子包初始化
│       │   ├── protocol_mode.py              # Protocol Mode：StateRef 链路、AMP 日志、MemoryUnit 写入
│       │   └── text_mode.py                  # Text Mode：长文本 baseline 和文本指标统计
│       │
│       ├── protocol/
│       │   ├── __init__.py                   # AMP 子包初始化
│       │   ├── capability.py                 # CapabilityDescriptor 能力描述结构
│       │   ├── codec.py                      # AMPMessage orjson 序列化/反序列化
│       │   ├── enums.py                      # MsgType 枚举
│       │   ├── router.py                     # ProtocolRouter：按 target_agent 路由消息
│       │   └── schema.py                     # AMPMessage 数据结构和协议校验
│       │
│       ├── prompts/
│       │   ├── __init__.py                   # Prompt 子包初始化
│       │   └── store.py                      # PromptTemplateStore：按 Agent 加载和渲染提示词
│       │
│       ├── runtime/
│       │   ├── __init__.py                   # Runtime 子包初始化
│       │   ├── agent.py                      # BaseAgent：hello、capability advertise、handle 接口
│       │   ├── event_bus.py                  # EventBus：轻量同步事件发布订阅
│       │   ├── orchestrator.py               # default_registry：注册四个默认 Agent
│       │   ├── registry.py                   # RuntimeContext、AgentRegistry、能力查找
│       │   └── scheduler.py                  # SyncScheduler：同步调度占位接口
│       │
│       ├── sandbox/
│       │   ├── __init__.py                   # Sandbox 子包初始化
│       │   ├── limits.py                     # SandboxLimits：超时与输出长度限制
│       │   └── runner.py                     # SandboxRunner：临时目录执行 Python 并返回结构化结果
│       │
│       ├── state/
│       │   ├── __init__.py                   # State 子包初始化
│       │   ├── embedding.py                  # HashEmbeddingEncoder 与 cosine_similarity
│       │   ├── lineage.py                    # State lineage 事件结构化输出
│       │   ├── refs.py                       # StateRef 创建和解析
│       │   ├── schema.py                     # StateType 与 StateRecord
│       │   └── store.py                      # StateStore：payload 落盘、SQLite index、JSONL lineage
│       │
│       └── storage/
│           ├── __init__.py                   # Storage 子包初始化
│           ├── jsonl.py                      # 统一 JSONL 读写工具
│           └── paths.py                      # RuntimePaths：统一管理 runs/latest 下所有路径
│
└── tests/
    ├── test_capability_handshake.py          # Agent 注册、能力发现、协议日志消息类型测试
    ├── test_config.py                        # `.env`/环境变量配置和密钥脱敏测试
    ├── test_embedding_ref.py                 # HashEmbedding 确定性、归一化和 StateStore 写入测试
    ├── test_memory_store.py                  # MemoryUnit 写入、关键词/标签/语义检索、评分策略测试
    ├── test_modes_and_benchmark.py           # Text/Protocol 双模式、Benchmark、Report 集成测试
    ├── test_interactive_agent.py             # 交互式 Agent 使用可配置 prompt 和 LLM client 测试
    ├── test_prompt_and_llm.py                # PromptTemplateStore 与 OpenAI-compatible client 测试
    ├── test_protocol_schema.py               # AMPMessage 字段校验和 codec roundtrip 测试
    ├── test_sandbox_runner.py                # SandboxRunner 成功执行和超时测试
    └── test_state_store.py                   # StateRef、StateStore、lineage、state_index 测试
```

## 核心运行流程

Protocol Mode 的主链路如下：

```text
读取任务文本
  ↓
写入 TextState，生成 state://text/...
  ↓
HashEmbedding 写入 EmbeddingState，生成 state://embedding/...
  ↓
PlannerAgent 生成计划并写 SummaryState
  ↓
RetrieverAgent 查询共享记忆并写 EvidenceState
  ↓
ExecutorAgent 调用 SandboxRunner 并写 CodeResultState
  ↓
SummarizerAgent 汇总结果并写 MemoryUnit
  ↓
输出 protocol/messages.jsonl、states.jsonl、memory.jsonl、trace.jsonl
```

Text Mode 的主链路如下：

```text
Planner -> Retriever -> Executor -> Summarizer
```

每一步都传递完整文本上下文，用于和 Protocol Mode 做通信开销对比。

## Benchmark 指标

`agentmesh benchmark` 会输出：

- `message_count`：Agent 间消息数量。
- `text_chars`：文本通信字符开销。
- `estimated_tokens`：粗略 token 估算。
- `protocol_bytes`：结构化协议消息字节数。
- `state_transfer_count`：状态传递次数。
- `state_transfer_bytes`：状态 payload 总字节数。
- `memory_query_count`：记忆查询次数。
- `memory_hit_count`：记忆命中次数。
- `memory_hit_rate`：记忆命中率。
- `latency_ms`：任务耗时。
- `answer_quality_score`：确定性质量估算。

汇总指标：

- `TokenSavingRate`
- `LatencyReductionRate`
- `MemoryHitRate`
- `QualityPreservationRate`

## openEuler 部署

```bash
bash scripts/setup_openeuler.sh
```

脚本会安装 Python、SQLite、gcc、make、git、curl、uv，并运行基础测试。详细说明见 [docs/openeuler_deploy.md](docs/openeuler_deploy.md)。

## 当前实现边界

- 默认 Agent 是确定性规则实现，用于保证离线可复现 Benchmark。
- `.env` 已支持大模型 endpoint/key/model 配置；`agentmesh chat` 会真实调用模型接口。
- Protocol Mode 中四个 Agent 在 LLM 配置可用时优先调用模型，调用失败会回退确定性逻辑。
- `Blob` state 有 API 支持，默认 demo 主要展示 text、embedding、summary、evidence、code_result。
- Dashboard/FastAPI 是第二阶段可选能力，当前未默认启用。
## Real-agent LLM handoff

`AGENTMESH_LLM_BASE_URL` uses an OpenAI-compatible Chat Completions endpoint. It can be either the API root, for example `https://api.example.com/v1`, or a full `/chat/completions` URL; the client normalizes it automatically.

After `.env` is configured, Protocol Mode becomes a real model-backed multi-agent run:

```bash
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
```

The handoff is:

```text
PlannerAgent LLM output -> SummaryState(plan_ref)
RetrieverAgent LLM output -> EvidenceState(evidence_ref)
ExecutorAgent LLM validation -> CodeResultState(code_result_ref)
SummarizerAgent LLM output -> final SummaryState + MemoryUnit + CLI answer
```

If an LLM call fails, the affected agent falls back to deterministic local behavior so offline benchmark runs remain reproducible.

## Contest comparison target

The benchmark compares two agent communication systems on the same tasks:

- `Text Mode`: traditional baseline. Each agent sends the full accumulated text context to the next agent.
- `Protocol Mode`: new system. Agents send structured AMP messages plus compact `state://...` refs; payloads live in `StateStore`, and Rust Core can accelerate codec, StateRef parsing, hash embedding, semantic top-k, and sandbox subprocess primitives.

The benchmark summary now includes `WireBytesReductionRate`, `TextWireBytes`, `ProtocolWireBytes` (compact `STATE_REF` handoff payload), `ProtocolCompactMessageBytes` (full compact AMP transport), `ProtocolJsonWireBytes` (readable protocol log size), `ProtocolStatePayloadBytes`, `RustCoreEnabledRuns`, and `RustSandboxBackendRuns` in addition to token, latency, memory, and quality metrics.
