# AgentMesh-Runtime

AgentMesh-Runtime 是一个面向多 Agent 协作的轻量级运行时原型。它的核心目标不是再做一个普通聊天机器人，而是验证一套可落地的协作机制：用结构化协议、状态引用、共享记忆、可选 Rust 热路径和可控工具执行，降低多 Agent 在连续任务中的重复上下文传递成本。

项目根目录是 `E:/system-compute`。第一版保持离线可运行，默认不依赖外部模型、GPU、向量数据库、消息队列、LangChain、numpy 或 torch；配置大模型和 TEI embedding 后，可以切换到真实 Agent 互动。

## 当前实现概览

已经实现的主线能力：

- 四个默认 Agent：`PlannerAgent`、`RetrieverAgent`、`ExecutorAgent`、`SummarizerAgent`。
- 两套可公平对比的运行模式：Text Mode 和 Protocol Mode。
- AMP 协议消息、能力声明、握手、协议映射、orjson/msgpack 编解码。
- Typed Envelope：会话字典压缩、结构化 payload 分离和字节开销统计。
- StateStore：`TextState`、`EmbeddingState`、`EvidenceState`、`CodeResultState`、`SummaryState`、`BlobState`，Agent 之间只传 `state://...` 引用。
- Embedding：默认 `HashEmbeddingEncoder` 离线运行，可选 Hugging Face TEI 服务。
- Shared Memory：当前 run 记忆和全局长期记忆合并检索，支持关键词、标签、语义检索和生命周期维护。
- CodeAct：Executor 可生成 Python，交给 `SandboxRunner` 执行，并写入 `CodeResultState`。
- 交互式 CLI：`agentmesh shell`、`agentmesh chat`、`agentmesh compare`、`agentmesh memory`、`agentmesh trace` 等。
- Benchmark 和报告：连续任务套件、长上下文任务套件、summary/detail/report 产物。
- 可选 Rust Core：codec、StateRef、embedding、top-k、typed envelope、sandbox subprocess 等热路径可由 `agentmesh_core` 加速。

## 公平对比边界

为了让评测结果能说明“通信机制”的差异，项目明确区分两套系统：

Text Mode 是传统 baseline。Agent 按固定链路协作，每次 handoff 都传递完整累积文本上下文，不使用 AMP、Typed Envelope、StateRef、Embedding、共享记忆、沙箱或 Rust Core。

Protocol Mode 是 AgentMesh 方案。Agent 通过 AMP 结构化消息协作，payload 落入 StateStore，handoff 中主要传递 `state://...` 引用；运行时可使用 Typed Envelope、共享记忆、CodeAct、动态路由、反馈回合和可选 Rust Core。

Benchmark 默认关闭 LLM，保证可复现；使用 `--llm` 时可以验证真实模型 Agent，但这时结果会受模型、网络和服务端状态影响。

## 架构设计

整体分层如下：

```text
CLI / Shell
  -> modes/text_mode.py              # 传统完整文本协作 baseline
  -> modes/protocol_mode.py          # AgentMesh 协议协作闭环
       -> runtime/scheduler.py       # 根据 capability 路由 Agent
       -> protocol/*.py              # AMP、codec、typed envelope、transport
       -> state/*.py                 # StateRef、StateStore、lineage、embedding
       -> memory/*.py                # SQLite FTS、语义检索、长期记忆
       -> sandbox/*.py               # CodeAct Python 执行
       -> agents/*.py                # Planner/Retriever/Executor/Summarizer
       -> llm/client.py              # OpenAI-compatible 模型接口
       -> prompts/store.py           # 可配置初始提示词模板
  -> eval/*.py                       # 指标、benchmark、报告
  -> crates/agentmesh-core           # 可选 Rust/PyO3 热路径
```

Protocol Mode 的主流程：

1. 初始化 `RuntimeContext`，加载 `.env`、prompt 模板、路径和 trace id。
2. 写入 `HELLO`、`CAPABILITY_ADVERTISE`、`CAPABILITY_QUERY`、`PROTOCOL_MAP`。
3. 将用户任务写成 `TextState`，将查询向量写成 `EmbeddingState`。
4. Planner 进行意图识别，输出 `PlannerDecision`，决定是否需要检索、工具执行和总结。
5. Retriever 从 run 记忆和全局长期记忆中检索 evidence，写入 `EvidenceState`。
6. Executor 在需要工具时生成 Python，Sandbox 执行后写入 `CodeResultState`。
7. 工具反馈最多触发一轮 Planner refine 和 Retriever refine。
8. Summarizer 读取结构化 state context，生成 `SummaryState`。
9. Memory tagger 将本轮结论转为 `MemoryUnit`，按写入策略进入当前 run 记忆和长期记忆。
10. 汇总 trace、状态日志、协议日志和 benchmark 指标。

## 关键机制

### AMP 协议

`AMPMessage` 使用 Pydantic schema 校验消息。关键字段包括：

- `trace_id`：一次任务链路的追踪 id。
- `source_agent` / `target_agent`：消息发送方和接收方。
- `msg_type`：`HELLO`、`CAPABILITY_ADVERTISE`、`CAPABILITY_QUERY`、`PROTOCOL_MAP`、`INVOKE`、`RESULT`、`ERROR` 等。
- `action`：能力动作，例如 `plan.create`、`memory.semantic_search`、`tool.run_python`、`summary.create`。
- `params` / `result`：结构化参数和结果。
- `state_refs`：跨 Agent 传递的状态引用。
- `capability`：能力声明和握手信息。

### Typed Envelope

Protocol Mode 保留可读的 AMP JSONL 日志，同时额外统计低开销 envelope：

```text
session dictionary: trace/agent/msg_type/action/capability/state_ref -> integer id
typed envelope:     {i, q, s, t, m, a?, c?, r?, p?}
payload store:      params/result payload 独立编号，由 p 引用
```

这避免在每条 handoff 中重复传递长字段名、Agent 名、action 名、capability 列表和完整 `state://...` 字符串。Python 会优先调用 Rust Core 编解码；未安装扩展时回退到 Python/msgpack/orjson。

### StateStore 和 StateRef

StateStore 将大 payload 从消息中剥离，写入 `runs/latest/data/states/` 或共享内存，并用 SQLite 建索引。Agent 之间传递的是引用：

```text
state://text/<state_id>
state://embedding/<state_id>
state://evidence/<state_id>
state://code_result/<state_id>
state://summary/<state_id>
```

每条 `StateRecord` 记录 producer、consumer、parent refs、payload 大小、metadata、是否写入记忆等信息，因此可以追踪 lineage。配置 `AGENTMESH_STATE_PAYLOAD_BACKEND=shm` 后，大于阈值的 payload 会走 Python shared memory，并统计 `state_shm_transfer_count` 和 `state_shm_transfer_bytes`。

### Embedding

默认离线方案是 `HashEmbeddingEncoder`：分词后加入 bigram 特征，用 SHA-256 映射到固定维度桶，带符号累加并 L2 归一化。它可复现、零依赖，适合比赛 MVP 和单元测试。

可选 TEI 方案通过 `AGENTMESH_EMBEDDING_PROVIDER=tei` 启用，调用 `/embed` 接口。TEI 不可用时会自动回退到 HashEmbedding，保证 Protocol Mode 仍可运行。

### 共享记忆

记忆分两层：

- 当前 run 记忆：`runs/latest/data/memory.sqlite`，随 run 产物生成。
- 全局长期记忆：`data/agentmesh_memory.sqlite`，不会因为清理 `runs/latest` 而丢失。

`HybridMemoryStore` 会合并两层检索结果并去重。检索方式包括：

- 关键词检索：SQLite FTS5。
- 标签检索：匹配 `MemoryUnit.tags`。
- 语义检索：query embedding 与 memory embedding 做 cosine similarity，再结合有效性、置信度、复用次数、时间衰减和标签重合度排序。

写入策略由 `MemoryWritePolicy` 控制：摘要长度、confidence、validity_score 达标后才写入；长期记忆还要求 importance 和 validity 达标，并过滤 transient/debug_noise。生命周期维护会按年龄、复用次数和 importance 归档低价值记忆。

### 动态路由和反馈

`PlannerDecision` 会根据用户意图和模型输出归一化出执行路线：

```text
analysis route: PlannerAgent -> RetrieverAgent -> SummarizerAgent
tool route:     PlannerAgent -> RetrieverAgent -> ExecutorAgent -> SummarizerAgent
```

如果任务不需要工具，Executor 会被跳过；如果需要 CodeAct，执行结果会以结构化 tool feedback 返回，最多触发一轮 Planner 和 Retriever refine。指标中会记录 `dynamic_route`、`selected_agents`、`skipped_agents`、`feedback_round_count`、`planner_refine_count`、`retriever_refine_count` 和 `tool_feedback_count`。

### CodeAct 和 Sandbox

Executor 在确定性模式下生成可复现 Python 校验代码；启用 LLM 后，会要求模型只返回 Python 代码。`SandboxRunner` 会加一层 Python guard，限制越界文件访问、网络和 subprocess，并支持三种后端：

- `rust`：安装 Rust Core 后优先使用 Rust subprocess runner。
- `python`：普通 Python subprocess 回退。
- `warm_python_unsafe`：可选热 worker，用于实验启动开销，不作为默认安全边界。

Sandbox 输出 `stdout`、`stderr`、`exit_code`、`latency_ms` 和 backend 信息，并写入 `CodeResultState`。

### LLM 和提示词模板

真实可互动 Agent 通过 OpenAI-compatible Chat Completions 接口接入。只要 `.env` 中配置了 base URL、API key 和 model，`--llm`、`chat`、`shell /ask` 就可以调用模型。

初始提示词模板单独存储在 `prompts/`：

```text
prompts/planner.md
prompts/retriever.md
prompts/executor.md
prompts/summarizer.md
prompts/interactive.md
```

也可以用 `AGENTMESH_PROMPT_DIR` 指向自定义模板目录。模板通过 `{task}`、`{query}`、`{input}`、`{user_input}` 等变量渲染；缺失模板时会使用内置 fallback。

## 指标计算方式

核心指标来自 `src/agentmesh/eval/metrics.py` 和 `src/agentmesh/eval/benchmark.py`。

### 单次运行指标

- `message_count`：当前模式写出的消息数量。
- `text_chars`：参与 handoff 的文本字符量。
- `estimated_tokens`：估算 token 数。当前估算器为 `mixed_cjk`：中文/CJK 字符按 `0.67` token 计，其他字符按 `1/4` token 计，结果至少为 1。
- `text_wire_bytes`：Text Mode 完整文本 handoff 的 UTF-8 字节数。
- `protocol_bytes`：Protocol Mode 可读 AMP JSONL 消息总字节数。
- `wire_bytes`：当前模式用于对比的总 wire bytes。Text Mode 使用 `text_wire_bytes`；Protocol Mode 使用结构化 envelope、transport、状态引用和 payload 统计后的通信字节。
- `structured_handoff_bytes`：Protocol Mode 中 handoff 结构化消息的字节量。
- `session_dictionary_bytes`：Typed Envelope 会话字典字节量。
- `typed_envelope_bytes`：压缩 envelope 字节量。
- `typed_payload_bytes`：payload store 字节量。
- `compact_structured_message_bytes`：更紧凑结构化消息估算字节量。
- `state_transfer_count`：本次 trace 产生和传递的 state 数量。
- `state_transfer_bytes`：state payload 总字节量。
- `memory_query_count`：记忆检索次数。
- `memory_hit_count`：检索返回的 memory unit 数量。
- `memory_query_hit_count`：至少命中一条记忆的查询次数。
- `memory_reused_unit_count`：被复用并增加 reuse_count 的记忆数量。
- `latency_ms`：端到端耗时。
- `stage_latency_ms`：各阶段耗时拆分。
- `answer_quality_score`：确定性质量估计，空答案为 0，非空答案从 0.6 起，随长度增加到最高 1.0。
- `transport_*`：transport 发送次数、字节数、平均延迟、p99 延迟。
- `state_shm_*`：shared memory payload 次数和字节数。

### Benchmark 聚合指标

Benchmark 会对同一任务分别运行 Text Mode 和 Protocol Mode，再按总量计算：

```text
TokenSavingRate = (TextEstimatedTokens - ProtocolEstimatedTokens) / TextEstimatedTokens
WireBytesReductionRate = (TextWireBytes - ProtocolWireBytes) / TextWireBytes
LatencyReductionRate = (TextLatencyMs - ProtocolLatencyMs) / TextLatencyMs
MemoryHitRate = MemoryQueryHitCount / MemoryQueryCount
QualityPreservationRate = ProtocolQualityScore / TextQualityScore
```

当分母为 0 时，对应 rate 记为 0。`MemoryHitRate` 会限制在 `[0, 1]`。Benchmark summary 还会汇总 Rust Core 启用次数、Rust sandbox 次数、transport 指标、shared memory 指标和反馈回合指标。

## 快速开始

```bash
uv sync --all-extras
uv run agentmesh --help
uv run agentmesh init
```

运行同一个任务的两种模式：

```bash
uv run agentmesh run --mode text --task examples/tasks/A1_requirements.txt
uv run agentmesh run --mode protocol --task examples/tasks/A1_requirements.txt
```

直接输入用户任务并对比两套系统：

```bash
uv run agentmesh compare "分析 AgentMesh Runtime 如何减少多 Agent 协作中的重复上下文传递"
```

允许两套系统调用你在 `.env` 中配置的模型：

```bash
uv run agentmesh compare "生成一个多步骤评测方案" --llm
```

单轮聊天：

```bash
uv run agentmesh chat --message "解释 AgentMesh 的 StateRef 机制"
```

## 交互式 Shell

启动：

```bash
uv run agentmesh shell
```

Shell 中直接输入一段任务文本，默认执行 `/compare`。常用命令：

```text
/help
/compare [--llm] <task>
/ask <message>
/run text|protocol <task-file>
/benchmark standard
/benchmark long
/memory --keyword|--tag|--semantic <query>
/trace [limit]
/report
/config
/exit
```

`/ask` 会把当前问题和 shell 历史写成临时 task，然后调用 Protocol Mode，因此会经过 Planner、Retriever、Executor、Summarizer、AMP、StateRef、Memory 和 trace 链路；`/compare` 仍按公平边界同时跑 Text Mode 和 Protocol Mode。

## Benchmark 和报告

标准连续任务：

```bash
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
uv run agentmesh report --run runs/latest
```

长上下文连续任务：

```bash
uv run agentmesh benchmark --suite examples/benchmarks/long_context_tasks.yaml
uv run agentmesh report --run runs/latest
```

运行后主要产物：

```text
runs/latest/
  benchmark_summary.csv       # 汇总指标
  benchmark_detail.jsonl      # 每轮 text/protocol 明细
  experiment_report.md        # 可提交的实验报告
  data/
    memory.sqlite             # 当前 run 记忆库
    state_index.sqlite        # StateRecord 索引
    states/                   # state payload 文件
  protocol/
    agent_io.jsonl             # Protocol Mode 每个 Agent 的输入、输出、state refs 和 action
    messages.jsonl            # AMP 协议消息
    memory.jsonl              # 本轮记忆写入日志
    states.jsonl              # 状态日志和 lineage
    trace.jsonl               # Protocol trace
  sandbox/                    # CodeAct 临时执行目录
  text/
    agent_io.jsonl             # Text Mode 每个 Agent 收到的全文上下文和输出
    messages.jsonl            # Text Mode 消息
    trace.jsonl               # Text Mode trace
```

## 大模型和环境配置

本地密钥写在 `.env` 中，仓库通过 `.gitignore` 忽略 `.env` 和 `.env.*`，只保留安全模板 `.env.example`。你已经在 `.env` 中配置模型接口后，可以直接使用 `--llm`、`chat` 或 `shell /ask`。

常用变量：

```bash
AGENTMESH_LLM_BASE_URL=https://api.example.com/v1
AGENTMESH_LLM_API_KEY=sk-...
AGENTMESH_LLM_MODEL=your-model
AGENTMESH_LLM_TIMEOUT_SECONDS=30
AGENTMESH_TEXT_LLM_TIMEOUT_SECONDS=30
AGENTMESH_PROMPT_DIR=
```

兼容变量：

```bash
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=your-model
```

Embedding 和 state 可选变量：

```bash
AGENTMESH_EMBEDDING_PROVIDER=hash
AGENTMESH_EMBEDDING_BASE_URL=http://127.0.0.1:8080
AGENTMESH_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
AGENTMESH_EMBEDDING_DIMENSIONS=512
AGENTMESH_EMBEDDING_TIMEOUT_SECONDS=15
AGENTMESH_STATE_PAYLOAD_BACKEND=file
AGENTMESH_STATE_SHM_THRESHOLD_BYTES=4096
AGENTMESH_MEMORY_MAINTENANCE_ENABLED=false
```

## TEI Embedding Docker

启动本地 TEI 服务：

```bash
docker compose -f docker-compose.embedding.yml up -d
```

测试：

```bash
curl http://127.0.0.1:8080/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs":"多 Agent 系统如何减少重复上下文传递？"}'
```

启用：

```bash
AGENTMESH_EMBEDDING_PROVIDER=tei
AGENTMESH_EMBEDDING_BASE_URL=http://127.0.0.1:8080
AGENTMESH_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
AGENTMESH_EMBEDDING_DIMENSIONS=512
```

## Rust Core

安装 Rust 工具链后，可以启用 PyO3 扩展：

```bash
uv sync --extra dev
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
```

Windows PowerShell 如果存在 Conda 环境变量冲突：

```powershell
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
```

未安装 Rust Core 时，Python 实现会自动接管，功能仍可运行。

## Docker 和 openEuler

普通容器：

```bash
docker compose up --build
```

openEuler 脚本：

```bash
bash scripts/setup_openeuler.sh
bash scripts/run_demo.sh
bash scripts/run_tests.sh
bash scripts/deploy_openeuler_docker.sh
```

部署说明见 `docs/openeuler_deploy.md` 和 `docs/docker_deploy.md`。

## 文件目录架构注释

下面按当前项目源文件列出每个文件的作用。`runs/`、`data/`、`.venv/`、`target/`、缓存目录和 `.env` 是本地运行产物或本地密钥，不应提交。

```text
E:/system-compute/
  .agent.md                                      # 本项目给 Agent/Codex 的本地开发约束和协作说明
  .dockerignore                                  # Docker 构建时排除缓存、运行产物和本地文件
  .env.example                                   # 可提交的环境变量模板，不包含真实密钥
  .gitignore                                     # 忽略 .env、runs、data、缓存、虚拟环境和构建产物
  .python-version                                # 本地 Python 版本提示
  2026-06-08-ipc-wasm-ebpf-enhancement-plan.md   # 后续 IPC/WASM/eBPF 增强规划
  AgentMesh_Runtime_项目落地计划书.md              # 原始落地计划和验收依据
  Cargo.lock                                     # Rust workspace 锁定依赖版本
  Cargo.toml                                     # Rust workspace 根配置
  Dockerfile                                     # AgentMesh Runtime 容器镜像构建文件
  Makefile                                       # 一键 lint、typecheck、test、benchmark、report 的质量入口
  README.md                                      # 当前项目说明、架构、指标、命令和文件注释
  docker-compose.embedding.yml                   # Hugging Face TEI embedding 服务编排
  docker-compose.yml                             # AgentMesh Runtime Docker Compose 编排
  pyproject.toml                                 # Python 包、依赖、CLI entrypoint、ruff、mypy、pytest 配置
  uv.lock                                        # uv 锁定依赖版本
  赛题.txt                                        # 比赛题目和约束材料

  crates/agentmesh-core/Cargo.toml               # Rust Core crate 配置
  crates/agentmesh-core/pyproject.toml           # maturin/PyO3 Python 扩展构建配置
  crates/agentmesh-core/src/codec.rs             # Rust 侧 orjson/msgpack/typed codec 热路径
  crates/agentmesh-core/src/embedding.rs         # Rust 侧 hash embedding、cosine、top-k 计算
  crates/agentmesh-core/src/envelope.rs          # Rust 侧 typed envelope 编解码
  crates/agentmesh-core/src/lib.rs               # PyO3 模块导出入口
  crates/agentmesh-core/src/sandbox_pool.rs      # Rust subprocess sandbox runner
  crates/agentmesh-core/src/state_ref.rs         # Rust 侧 StateRef 解析和校验
  crates/agentmesh-core/src/vector_index.rs      # Rust 侧向量索引和 memory rank top-k

  docs/architecture.md                           # 总体架构文档
  docs/benchmark_design.md                       # Benchmark 设计、任务套件和指标说明
  docs/docker_deploy.md                          # Docker 部署说明
  docs/implementation_audit.md                   # 实现覆盖审计记录
  docs/memory_design.md                          # 共享记忆模型和检索设计
  docs/openeuler_deploy.md                       # openEuler 部署说明
  docs/progress.md                               # 阶段进度、验收命令和结果记录
  docs/protocol_spec.md                          # AMP 协议字段、消息类型和校验规则
  docs/state_lifecycle.md                        # StateRecord 生命周期和 lineage 说明
  docs/superpowers/plans/2026-06-04-rust-runtime-core-refactor.md # Rust Core 重构计划
  docs/superpowers/plans/2026-06-08-ipc-shm-benchmark-lite.md     # IPC/SHM benchmark 轻量计划

  examples/benchmarks/continuous_tasks.yaml      # 标准连续任务 benchmark suite
  examples/benchmarks/long_context_tasks.yaml    # 长上下文连续任务 benchmark suite
  examples/corpora/contest_requirements.md       # 比赛要求示例语料
  examples/corpora/sample_repo_summary.md        # 示例仓库摘要语料
  examples/tasks/A1_requirements.txt             # A 组第 1 轮需求任务
  examples/tasks/A2_architecture.txt             # A 组第 2 轮架构任务
  examples/tasks/A3_modules.txt                  # A 组第 3 轮模块任务
  examples/tasks/A4_readme.txt                   # A 组第 4 轮文档任务
  examples/tasks/A5_report.txt                   # A 组第 5 轮报告任务
  examples/tasks/B1_repo_analysis.txt            # B 组第 1 轮仓库分析任务
  examples/tasks/B2_tests.txt                    # B 组第 2 轮测试任务
  examples/tasks/B3_bugfix.txt                   # B 组第 3 轮修复任务
  examples/tasks/B4_api_docs.txt                 # B 组第 4 轮 API 文档任务
  examples/tasks/B5_perf_report.txt              # B 组第 5 轮性能报告任务
  examples/tasks/long_context_A1.txt             # 长上下文 A 组第 1 轮
  examples/tasks/long_context_A2.txt             # 长上下文 A 组第 2 轮
  examples/tasks/long_context_A3.txt             # 长上下文 A 组第 3 轮
  examples/tasks/long_context_A4.txt             # 长上下文 A 组第 4 轮
  examples/tasks/long_context_A5.txt             # 长上下文 A 组第 5 轮
  examples/tasks/long_context_B1.txt             # 长上下文 B 组第 1 轮
  examples/tasks/long_context_B2.txt             # 长上下文 B 组第 2 轮
  examples/tasks/long_context_B3.txt             # 长上下文 B 组第 3 轮
  examples/tasks/long_context_B4.txt             # 长上下文 B 组第 4 轮
  examples/tasks/long_context_B5.txt             # 长上下文 B 组第 5 轮

  prompts/executor.md                            # ExecutorAgent 初始提示词模板
  prompts/interactive.md                         # chat 命令使用的交互助手提示词模板
  prompts/planner.md                             # PlannerAgent 初始提示词模板
  prompts/retriever.md                           # RetrieverAgent 初始提示词模板
  prompts/summarizer.md                          # SummarizerAgent 初始提示词模板

  scripts/deploy_openeuler_docker.sh             # openEuler Docker 部署脚本
  scripts/profile_protocol.py                    # 协议开销 profiling 脚本
  scripts/run_demo.sh                            # Demo 运行脚本
  scripts/run_tests.sh                           # 测试运行脚本
  scripts/setup_openeuler.sh                     # openEuler 环境初始化脚本

  src/agentmesh/__init__.py                      # Python 包版本和顶层导出
  src/agentmesh/cli.py                           # Typer CLI 命令入口
  src/agentmesh/config.py                        # .env/env 配置加载和 Pydantic 配置模型
  src/agentmesh/core.py                          # 可选 Rust Core 探测和懒加载
  src/agentmesh/errors.py                        # 项目统一异常类型

  src/agentmesh/agents/__init__.py               # 默认 Agent 包导出
  src/agentmesh/agents/executor.py               # ExecutorAgent，生成 CodeAct Python 或确定性执行输入
  src/agentmesh/agents/planner.py                # PlannerAgent，意图识别和执行路线规划
  src/agentmesh/agents/retriever.py              # RetrieverAgent，记忆检索和 evidence 构建
  src/agentmesh/agents/summarizer.py             # SummarizerAgent，最终回答和记忆摘要生成

  src/agentmesh/chat/__init__.py                 # chat 包导出
  src/agentmesh/chat/session.py                  # chat 命令的单轮/历史对话逻辑

  src/agentmesh/eval/__init__.py                 # eval 包导出
  src/agentmesh/eval/benchmark.py                # Benchmark suite 读取、双模式运行和聚合
  src/agentmesh/eval/compare.py                  # 用户 prompt 双模式即时对比
  src/agentmesh/eval/metrics.py                  # RunMetrics、token 估算和指标模型
  src/agentmesh/eval/quality.py                  # 确定性答案质量估计
  src/agentmesh/eval/report.py                   # experiment_report.md 生成

  src/agentmesh/llm/__init__.py                  # LLM 包导出
  src/agentmesh/llm/client.py                    # OpenAI-compatible Chat Completions 客户端

  src/agentmesh/memory/__init__.py               # memory 包导出
  src/agentmesh/memory/hybrid_store.py           # 当前 run 记忆和全局记忆的合并检索
  src/agentmesh/memory/lifecycle.py              # 记忆保留、归档和 importance 决策
  src/agentmesh/memory/maintenance.py            # 后台记忆维护 worker
  src/agentmesh/memory/policy.py                 # 记忆写入和长期写入策略
  src/agentmesh/memory/schema.py                 # MemoryUnit 数据结构
  src/agentmesh/memory/scorer.py                 # 语义、有效性、复用、时效、标签综合评分
  src/agentmesh/memory/search.py                 # MemorySearchResult 结果模型
  src/agentmesh/memory/sqlite_store.py           # SQLite/FTS5 记忆存储和检索实现
  src/agentmesh/memory/tagger.py                 # LLM/规则记忆分类和标签生成

  src/agentmesh/modes/__init__.py                # modes 包导出
  src/agentmesh/modes/protocol_mode.py           # Protocol Mode 完整闭环实现
  src/agentmesh/modes/text_mode.py               # Text Mode baseline 实现

  src/agentmesh/prompts/__init__.py              # prompts 包导出
  src/agentmesh/prompts/store.py                 # 文件化 prompt 模板加载和变量渲染

  src/agentmesh/protocol/__init__.py             # protocol 包导出
  src/agentmesh/protocol/capability.py           # Agent 能力声明、匹配和握手构造
  src/agentmesh/protocol/codec.py                # AMP 消息 orjson/msgpack 编解码
  src/agentmesh/protocol/enums.py                # MsgType 枚举
  src/agentmesh/protocol/envelope.py             # Typed Envelope 会话字典和字节统计
  src/agentmesh/protocol/router.py               # 协议路由辅助逻辑
  src/agentmesh/protocol/schema.py               # AMPMessage Pydantic schema 和校验
  src/agentmesh/protocol/transport.py            # InProc/Socket frame transport 和 transport 指标

  src/agentmesh/runtime/__init__.py              # runtime 包导出
  src/agentmesh/runtime/agent.py                 # BaseAgent 抽象和 handle 接口
  src/agentmesh/runtime/decision.py              # PlannerDecision 和动态路由归一化
  src/agentmesh/runtime/event_bus.py             # 轻量事件总线
  src/agentmesh/runtime/orchestrator.py          # 同步 orchestrator 基础设施
  src/agentmesh/runtime/registry.py              # AgentRegistry、CapabilityRegistry、RuntimeContext
  src/agentmesh/runtime/scheduler.py             # ProtocolScheduler，按 action/capability 调度 Agent

  src/agentmesh/sandbox/__init__.py              # sandbox 包导出
  src/agentmesh/sandbox/limits.py                # 沙箱 timeout 和输出长度限制
  src/agentmesh/sandbox/runner.py                # Python/Rust/warm worker 沙箱执行器
  src/agentmesh/sandbox/worker.py                # warm Python worker 子进程入口

  src/agentmesh/shell/__init__.py                # shell 包导出
  src/agentmesh/shell/commands.py                # slash command 解析
  src/agentmesh/shell/render.py                  # Rich 终端渲染
  src/agentmesh/shell/session.py                 # 交互式 shell 会话和命令分发

  src/agentmesh/state/__init__.py                # state 包导出
  src/agentmesh/state/embedding.py               # Hash/TEI embedding encoder 和 cosine similarity
  src/agentmesh/state/lineage.py                 # State lineage 构建
  src/agentmesh/state/refs.py                    # StateRef 解析和格式化
  src/agentmesh/state/schema.py                  # StateRecord、StateType、StateRef schema
  src/agentmesh/state/store.py                   # State payload 落盘/SHM、SQLite 索引和 consumer 记录

  src/agentmesh/storage/__init__.py              # storage 包导出
  src/agentmesh/storage/jsonl.py                 # JSONL 读写工具
  src/agentmesh/storage/paths.py                 # RuntimePaths 路径集中管理

  tests/test_capability_handshake.py             # capability handshake 单元测试
  tests/test_compare_prompt.py                   # compare prompt 双模式测试
  tests/test_config.py                           # 配置加载测试
  tests/test_core_optional.py                    # Rust Core 可选加载测试
  tests/test_dynamic_agent_routing.py            # 动态路由测试
  tests/test_embedding_ref.py                    # EmbeddingRef/StateRef 测试
  tests/test_hybrid_memory.py                    # HybridMemoryStore 测试
  tests/test_interactive_agent.py                # 交互式 Agent/LLM 路径测试
  tests/test_memory_rust_vector_index.py         # Rust vector index 兼容测试
  tests/test_memory_store.py                     # SQLiteMemoryStore 测试
  tests/test_metrics.py                          # 指标计算测试
  tests/test_modes_and_benchmark.py              # text/protocol/benchmark 集成测试
  tests/test_prompt_and_llm.py                   # prompt 模板和 LLM client 测试
  tests/test_protocol_schema.py                  # AMP schema 和校验测试
  tests/test_real_agent_protocol_mode.py         # Protocol Mode 真实 Agent 链路测试
  tests/test_rust_codec_parity.py                # Rust/Python codec parity 测试
  tests/test_rust_embedding_parity.py            # Rust/Python embedding parity 测试
  tests/test_sandbox_runner.py                   # SandboxRunner 测试
  tests/test_shell.py                            # shell 命令测试
  tests/test_state_store.py                      # StateStore 测试
  tests/test_state_store_fast_index.py           # StateStore 快速索引测试
  tests/test_transport.py                        # transport frame 和指标测试
  tests/test_typed_envelope.py                   # Typed Envelope 测试
```

## 质量验证

常用质量门禁：

```bash
uv run ruff check .
uv run mypy src
uv run pytest
make all
```

`make all` 会串联 lint、typecheck、test、benchmark 和 report。文档修改不一定需要重新跑完整测试，但发布前建议至少跑一次。

## 当前边界

- Socket transport 已有 frame 实现和测试基础，但默认运行仍使用 in-process transport。
- Benchmark 默认关闭 LLM，主要衡量通信协议和状态传递机制；启用 `--llm` 后可用于真实交互验证。
- Sandbox 是轻量隔离和比赛原型，不等同于生产级容器、WASM 或硬隔离环境。
- HashEmbedding 是可复现离线检索基线，真实语义效果建议接入 TEI 或后续 embedding 服务。
