# AgentMesh-Runtime

## 公平对比约束

Text Mode 是传统 Agent 协作 baseline：一个 Agent 只把自己的完整文本上下文原样传给下一个 Agent，中间不使用 Rust Core、typed envelope、StateRef、embedding、共享记忆、沙箱或任何结构化状态处理。

Protocol Mode 才是新系统：使用 AMP 结构化消息、typed envelope、StateRef、embedding、共享记忆、CodeAct 沙箱和可选 Rust Core 热路径。所有 benchmark 和 `agentmesh compare` 都按这个边界比较两套系统。

AgentMesh-Runtime 是一个面向多 Agent 协作的轻量级运行时原型。项目重点不是做普通聊天机器人，而是验证一套从“自然语言长文本传递”转向“结构化协议 + StateRef 状态传递 + Rust 热路径”的系统机制。

系统同时提供两种运行模式：

- **Text Mode**：传统 baseline。Agent 之间直接传递完整累积文本上下文。
- **Protocol Mode**：新系统。Agent 之间传递 AMP 结构化消息、typed envelope、`state://...` 引用和共享记忆，非文本 payload 存入 StateStore。

## 赛题要求覆盖

- 不少于 3 个 Agent：内置 `PlannerAgent`、`RetrieverAgent`、`ExecutorAgent`、`SummarizerAgent`。
- 角色覆盖：任务规划、信息检索、工具执行、总结生成。
- 结构化通信：`AMPMessage` 包含 `action`、`params`、`result`、`capability`、`state_refs`。
- 握手和能力发现：支持 `HELLO`、`CAPABILITY_ADVERTISE`、`CAPABILITY_QUERY`、`PROTOCOL_MAP`。
- 双模式对比：`agentmesh run --mode text|protocol` 和 `agentmesh compare`。
- 非文本状态传递：`HashEmbeddingEncoder` 生成 `EmbeddingState`，通过 `state://embedding/...` 传递。
- 共享记忆：`MemoryUnit` 存储记忆 ID、来源 Agent、创建时间、任务主题、摘要、标签、证据和 StateRef。
- 记忆检索：支持关键词、标签和语义相似度检索。
- 连续任务：`examples/benchmarks/continuous_tasks.yaml` 和 `long_context_tasks.yaml` 均包含 A/B 两组关联任务。
- 评测指标：消息数、token/字符开销、wire bytes、状态传递次数和规模、耗时、记忆命中率、质量保持率。
- CodeAct：`ExecutorAgent` 可让 LLM 生成 Python 代码，交给 `SandboxRunner` 执行，并将结果写入 `CodeResultState`。
- Rust Core：可选 `agentmesh_core` 扩展加速 StateRef、codec、msgpack、typed envelope、embedding、top-k 和 sandbox subprocess。

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

用户直接输入提示词，同时运行两套系统并比较指标：

```bash
uv run agentmesh compare "分析 AgentMesh Runtime 如何减少多 Agent 协作中的重复上下文传输"
```

如需让 Protocol Mode 中的 Agent 调用已配置的大模型：

```bash
uv run agentmesh compare "生成一个多步骤评测方案" --llm
```

## 交互式 CLI Shell

项目也提供类似命令行工具的交互入口：

```bash
uv run agentmesh shell
```

进入 shell 后，直接输入一段任务文本会默认执行 `compare`，也就是在相同任务条件下同时跑 Text Mode 和 Protocol Mode，并输出消息数、token、wire bytes、状态传递、记忆命中率等指标。

常用 `/` 命令：

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

这个 shell 只是复用现有运行时能力，不改变公平对比边界：普通文本协作仍然是完整文本直传，结构化协议协作才会使用 AMP、StateRef、embedding、共享记忆和 Rust Core。

## Benchmark

标准连续任务：

```bash
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
uv run agentmesh report --run runs/latest
```

长上下文连续任务，用于放大 StateRef 和 typed envelope 优势：

```bash
uv run agentmesh benchmark --suite examples/benchmarks/long_context_tasks.yaml
uv run agentmesh report --run runs/latest
```

输出指标包括：

- `TokenSavingRate`
- `WireBytesReductionRate`
- `TextWireBytes`
- `ProtocolWireBytes`
- `ProtocolSessionDictionaryBytes`
- `ProtocolTypedEnvelopeBytes`
- `ProtocolTypedPayloadBytes`
- `ProtocolCompactMessageBytes`
- `ProtocolJsonWireBytes`
- `ProtocolStatePayloadBytes`
- `LatencyReductionRate`
- `MemoryHitRate`
- `QualityPreservationRate`
- `RustCoreEnabledRuns`
- `RustSandboxBackendRuns`

## 大模型配置

项目默认可离线运行。需要真实 LLM Agent 时，复制配置模板：

```bash
cp .env.example .env
```

填写 OpenAI-compatible Chat Completions 接口：

```bash
AGENTMESH_LLM_BASE_URL=https://api.example.com/v1
AGENTMESH_LLM_API_KEY=sk-...
AGENTMESH_LLM_MODEL=your-model
AGENTMESH_LLM_TIMEOUT_SECONDS=30
AGENTMESH_PROMPT_DIR=
```

配置后可运行：

```bash
uv run agentmesh chat --message "解释 AgentMesh 的 StateRef 机制"
uv run agentmesh compare "让多 Agent 生成一个性能分析方案" --llm
```

LLM 调用失败时，Agent 会回退到确定性逻辑，保证 benchmark 可复现。

## TEI Embedding Docker

Protocol Mode 可以使用 Hugging Face Text Embeddings Inference 部署 `BAAI/bge-small-zh-v1.5`。Text Mode 仍然不使用 embedding。

启动 embedding 服务：

```bash
docker compose -f docker-compose.embedding.yml up -d
```

测试服务：

```bash
curl http://127.0.0.1:8080/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs":"多 Agent 系统如何减少重复上下文传输？"}'
```

`.env` 中启用 TEI：

```bash
AGENTMESH_EMBEDDING_PROVIDER=tei
AGENTMESH_EMBEDDING_BASE_URL=http://127.0.0.1:8080
AGENTMESH_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
AGENTMESH_EMBEDDING_DIMENSIONS=512
AGENTMESH_EMBEDDING_TIMEOUT_SECONDS=15
```

如果 TEI 服务不可用，Protocol Mode 会自动回退到本地 `HashEmbeddingEncoder`，确保 benchmark 仍可运行。

## CodeAct 执行链

Protocol Mode 的工具执行链路如下：

```text
RetrieverAgent -> EvidenceState
ExecutorAgent -> 生成 Python 代码
SandboxRunner -> 执行 Python
StateStore -> 写入 CodeResultState
SummarizerAgent -> 读取 CodeAct 结果摘要并生成最终 SummaryState
MemoryStore -> 写入 MemoryUnit
```

没有配置 LLM 时，`ExecutorAgent` 会生成一段确定性 Python 校验代码；配置 LLM 后，会提示模型只返回 Python 代码，并在轻量沙箱中执行。

## Typed Envelope 通信

Protocol Mode 保留可读 AMP JSON 日志，同时统计低开销 typed envelope：

```text
session dictionary: trace/agent/action/capability/msg_type/state_ref -> integer id
message envelope: {i, q, s, t, m, a?, c?, r?, p?}
payload store: params/result payloads addressed by p ids
```

这避免在 Agent handoff 中反复发送 agent 名、action 字符串、能力列表、完整 `state://...` 引用和大 payload。

Rust Core 已提供 typed envelope 编解码函数；Python 层优先调用 Rust，未安装新扩展时自动回退到 Python/msgpack 路径。

## Rust Core

安装 Rust 工具链后启用扩展：

```bash
uv sync --extra dev
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
```

Windows PowerShell 如果存在 Conda 环境变量冲突：

```powershell
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
uv run maturin develop --manifest-path crates/agentmesh-core/Cargo.toml
```

未安装 Rust Core 时，项目自动回退到纯 Python 实现。

## 共享记忆检索

Protocol Mode 会先写入当前 run 记忆，并通过 `importance_score`、`validity_score`、`confidence`、记忆类型和摘要质量判断是否同步进入全局长期记忆库。长期记忆库位于 `data/agentmesh_memory.sqlite`，不会因为清理 `runs/latest` 而丢失。

检索时会合并当前 run 记忆和全局长期记忆，并按语义相似度、有效性、置信度、复用次数、时间衰减和标签重合度综合排序。命中的历史记忆会作为结构化 evidence 注入后续 Agent 链路。

如果安装了 Rust Core，综合排序会优先调用 `memory_rank_top_k`，在 Rust 中批量计算 cosine similarity、metadata score 和 top-k；未安装或版本不完整时自动回退 Python 实现。

```bash
uv run agentmesh memory search --keyword protocol
uv run agentmesh memory search --tag protocol
uv run agentmesh memory search --semantic "agent state passing"
uv run agentmesh memory stats
uv run agentmesh memory archive
```

`agentmesh shell` 启动时会按配置启动低频后台维护线程，对全局长期记忆做轻量归档，不阻塞主要多 Agent 任务。

## 运行产物

命令执行后会生成 `runs/latest/`：

```text
runs/latest/
├── benchmark_summary.csv
├── benchmark_detail.jsonl
├── experiment_report.md
├── data/
│   ├── memory.sqlite
│   ├── state_index.sqlite
│   └── states/
├── protocol/
│   ├── messages.jsonl
│   ├── memory.jsonl
│   ├── states.jsonl
│   └── trace.jsonl
├── sandbox/
└── text/
    ├── messages.jsonl
    └── trace.jsonl
```

## 质量验证

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

openEuler 部署说明见 [docs/openeuler_deploy.md](docs/openeuler_deploy.md)。

## 当前边界

- Typed envelope 已有 Rust 编解码 API，但真实网络 IPC/Socket/shared memory transport 仍是后续增强方向。
- Benchmark 默认关闭 LLM，专注比较通信机制；使用 `--llm` 可测试真实模型 Agent。
- Sandbox 是轻量隔离，不等价于生产级容器或 WASM 沙箱。
