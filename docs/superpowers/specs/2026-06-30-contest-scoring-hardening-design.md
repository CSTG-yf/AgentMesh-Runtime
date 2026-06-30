# AgentMesh Runtime 竞赛冲分版设计方案

## 1. 目标

本轮打磨以第三届中国研究生操作系统开源创新大赛赛题评分项为验收标准，不以继续堆叠功能为目标。作品最终需要形成四层相互印证的证据：

1. 代码实现：多 Agent、结构化协议、非文本状态、共享记忆和评测链路真实运行。
2. 自动化测试：核心功能、异常路径和连续运行全部通过。
3. 原始实验：统一环境、重复运行、可追溯到每轮任务和每条消息。
4. 交付材料：报告、Dashboard、openEuler 验收记录和演示视频使用同一批数据。

目标评分与最低验收线：

| 评分项 | 目标分 | 最低验收线 |
|---|---:|---|
| 通信效率 | 20+/25 | Protocol 端到端 token 节省率不低于 30%，所有主实验为正值 |
| 状态传递创新 | 16+/20 | StateRef、EmbeddingRef、SHM/IPC 均有真实调用记录和消融对比 |
| 记忆复用效果 | 16+/20 | Warm 相比 Cold 的 token 或时延至少降低 20%，任务质量不下降 |
| 系统完整性 | 17+/20 | 全部测试通过，单次连续任务不少于 10 轮，故障后可恢复 |
| 实验验证 | 12+/15 | 相同环境重复不少于 5 次，报告均值、标准差、P50 和 P95 |
| 总目标 | 81+/100 | 达到可答辩、可复现、可解释的竞赛状态 |

## 2. 范围

### 2.1 保留

- Planner、Retriever、Executor、Summarizer 四类 Agent。
- AMP 结构化协议、Typed Envelope、StateRef 和 EmbeddingRef。
- SQLite 共享记忆、Rust Core、CodeAct 沙箱和静态 Dashboard。
- Text Mode 与 Protocol Mode 双模式。
- 离线确定性运行和真实 LLM 运行。
- AgentShell 作为主要交互入口，保留裸文本自动路由 `/ask`、逐 Agent
  流式输出、无损代码/Markdown 最终答案，以及 `/compare`、`/benchmark`、
  `/dashboard`、`/memory`、`/trace`、`/config` 和 `/exit` 命令体验。

### 2.2 重构

- Benchmark 配置、执行隔离、重复实验和统计汇总。
- token、wire bytes、payload bytes 和 end-to-end bytes 的统计边界。
- 回答质量评估，不再使用单纯按回答长度计分。
- 记忆复用决策，使高置信命中能够减少实际 Agent 调用或工具执行。
- 报告结构，使所有结论都能追溯到原始明细。

### 2.3 明确不做

- 不扩展为通用工作流平台。
- 不同时投入 eBPF、WASM、分布式向量数据库等多个高风险方向。
- 不为了得到正向指标而人为削弱 Text Mode。
- 不把离线实验和真实 LLM 实验混入同一个汇总值。

## 3. 总体架构

冲分版在现有运行时外增加一层“实验控制与证据”边界：

```text
Benchmark Controller
  ├─ Environment Manifest
  ├─ Dataset / Repeat / Seed Controller
  ├─ Text Mode Runner
  ├─ Protocol Cold Runner
  ├─ Protocol Warm Runner
  └─ Ablation Runner
        │
        ▼
AgentMesh Runtime
  ├─ Orchestrator / Scheduler
  ├─ AMP + Typed Envelope
  ├─ StateStore
  │    ├─ inline/file backend
  │    └─ shared-memory backend
  ├─ Shared Memory Store
  ├─ Planner / Retriever / Executor / Summarizer
  └─ Sandbox / Rust Core
        │
        ▼
Evidence Pipeline
  ├─ Raw run JSONL
  ├─ Metric validation
  ├─ Statistical aggregation
  ├─ Quality judge
  ├─ Markdown report
  └─ Static dashboard
```

Benchmark Controller 必须为每次运行生成唯一 `experiment_id`，并记录 Git commit、操作系统、Python/Rust 版本、模型配置摘要、随机种子、运行模式和 suite 版本。秘密信息不得进入 manifest。

## 4. 优先级与阶段

### P0：建立可信基线

目标是消除任何会让评委直接否定结果的问题。

1. 修复当前 7 项失败测试：
   - HELLO/CAPABILITY 消息缺失。
   - CodeAct payload schema 不一致。
   - 生成文件 metadata 不一致。
   - Rust memory rank 路径未执行。
   - 中文语义检索排序错误。
   - 真实 LLM executor state schema 不一致。
2. 将 160 项现有测试全部跑绿。
3. 固化 `RunMetrics` 字段语义和单位。
4. 增加 schema version，旧结果不得静默混入新报告。
5. 新增最少 10 轮连续运行测试，验证内存、状态、日志和沙箱不会相互污染。

P0 完成标准：

- `ruff`、`mypy`、`pytest` 全部通过。
- 连续 10 轮测试无失败、无残留锁、无跨轮错误复用。
- 当前功能行为没有依赖未记录的本地 `.env`。

### P1：重建公平的 Benchmark

目标是让通信效率结论经得住评审追问。

实验必须拆分为两个独立赛道：

- Deterministic Track：固定输出、固定 seed、无外部网络，用于协议和系统开销复现。
- LLM Track：固定模型、temperature=0、固定 prompt 版本，用于实际 Agent 效果验证。

两种模式必须使用：

- 相同任务输入。
- 相同可用角色。
- 相同工具能力。
- 相同质量验收。
- 相同冷启动或热启动条件。

Text Mode 可以传递完整文本，但不得故意重复与任务无关的内容。Protocol Mode 可以传递引用，但被引用 payload 的存储、读取和传输成本必须单独计量并进入端到端统计。

每个主 suite 至少包含 10 个逻辑任务，每个配置重复 5 次。任务顺序由固定 seed 派生，运行顺序采用 Text/Protocol 交替或配对随机化，降低缓存和服务抖动偏差。

### P2：统一指标口径

核心指标分成四组，禁止用一个含义模糊的 `wire_bytes_reduction_rate` 代替全部通信成本。

#### Agent 上下文成本

```text
agent_input_tokens  = 所有 Agent 输入 token 总和
agent_output_tokens = 所有 Agent 输出 token 总和
agent_total_tokens  = agent_input_tokens + agent_output_tokens
token_saving_rate   = 1 - protocol.agent_total_tokens / text.agent_total_tokens
```

真实 LLM 使用模型 tokenizer 或 API usage；离线实验使用固定 tokenizer。字符估算仅作为兼容指标，不能作为主结论。

#### 协议传输成本

```text
control_plane_bytes = envelope + dictionary + headers
payload_bytes       = inline payload + externalized payload 首次写入/发送
state_read_bytes    = 下游实际读取的 StateRef payload
wire_total_bytes    = control_plane_bytes + payload_bytes
```

持久化副本、日志副本和通信字节必须分栏，避免重复计算或选择性忽略。

#### 系统性能

- end-to-end latency。
- 各 Agent stage latency。
- memory search、state read/write、serialization、sandbox latency。
- P50、P95、均值、标准差和失败率。

#### 任务效果

- exact match、结构化断言或测试用例通过率。
- 必须执行的代码任务使用沙箱测试结果。
- 开放文本任务使用明确 rubric 的盲评或固定 judge。
- 质量保持率限定在可解释范围，不再由文本长度决定。

### P3：让记忆复用产生真实收益

现有链路只证明“查询命中”。新链路增加 `ReuseDecision`：

```text
Memory Search
  ├─ no hit / low confidence
  │    └─ full Planner -> Retriever -> Executor -> Summarizer
  ├─ reusable evidence
  │    └─ Planner -> reuse evidence -> Executor/Verifier -> Summarizer
  └─ reusable verified result
       └─ lightweight Verifier -> Summarizer
```

`ReuseDecision` 至少包含：

- memory ID。
- semantic similarity。
- tag/topic match。
- provenance task。
- artifact/test validity。
- reuse action。
- skipped stages。
- verification result。

只有通过来源、有效期和任务约束检查的记忆才能跳过阶段。代码结果必须重新执行轻量测试或校验 checksum，不能直接信任历史自然语言结论。

Warm/Cold 实验采用配对任务：

- Build：产生经过验证的策略、代码模板或证据。
- Warm：保留 Build 记忆执行关联任务。
- Cold：清空记忆执行与 Warm 完全相同的任务。

报告必须直接展示：

- Warm 与 Cold 的 Agent 调用数差异。
- 跳过的阶段。
- token、时延和工具执行次数差异。
- 质量差异。
- 错误复用率和拒绝复用率。

### P4：非文本状态与系统创新

创新点聚焦为“分层状态通道”，而不是继续增加互不关联的技术名词：

1. 小型 control message 使用 Typed Envelope。
2. 中型状态使用 StateRef + 文件/SQLite payload。
3. 超过阈值的大型 payload 使用 shared memory。
4. embedding 向量直接以二进制状态传递，不转写为自然语言。
5. Rust Core 负责 codec、向量 top-k 或沙箱热路径。

增加四组消融：

- Text full context。
- AMP + inline payload。
- AMP + StateRef。
- AMP + StateRef + SHM。

记录各组的 bytes、序列化耗时、端到端耗时和内存占用。SHM 只在超过阈值时启用，避免小 payload 上初始化成本反而拖慢系统。

### P5：报告、Dashboard 与交付

Dashboard 首页只展示评委需要的五类结论：

1. token 节省。
2. 端到端通信字节。
3. 时延与稳定性。
4. Warm/Cold 记忆收益。
5. 质量保持与失败率。

每个结论必须可下钻到：

- suite。
- task。
- repeat。
- mode。
- trace。
- 原始 metric record。

最终仓库增加受版本控制的 `artifacts/sample/`，存放一组体积受控、可复核的正式样例结果；`runs/` 继续作为本地临时目录。

正式交付包括：

- 技术报告。
- 系统设计文档。
- 部署与复现实验文档。
- 固定样例数据和图表。
- openEuler 验收日志。
- 5 至 8 分钟演示脚本。
- 演示视频分镜和数据口径说明。

## 5. Benchmark 套件设计

### 5.1 Protocol Efficiency

- 至少 10 个多阶段任务。
- 覆盖短、中、长三档上下文。
- 同时包含无需 Executor 和需要 CodeAct 的任务。
- 验证协议在不同 payload 尺寸下的收益边界。

### 5.2 Continuous Memory

- 至少 5 组 Build/Warm/Cold 三元任务。
- 每组任务共享策略或证据，但输入数据不同。
- Warm 不允许直接复用最终答案。
- 重点验证减少 Agent 调用、工具执行和重复生成。

### 5.3 State Transport Ablation

- 固定 payload 尺寸：1 KiB、16 KiB、256 KiB、1 MiB。
- 比较 inline、StateRef 和 SHM。
- 每档至少重复 20 次，报告微基准与端到端结果。

### 5.4 Reliability

- 连续执行不少于 10 轮。
- 注入一次无效记忆、一次沙箱失败、一次 payload 缺失。
- 验证系统拒绝错误复用、记录错误并继续后续任务。

## 6. 数据流与隔离

一次正式实验的数据流如下：

1. Controller 创建不可变 experiment manifest。
2. 为 Text、Protocol Cold、Protocol Warm 创建独立运行目录和数据库。
3. 根据 seed 生成配对运行顺序。
4. 每轮记录原始事件、Agent I/O、协议消息、状态访问和记忆决策。
5. Metric validator 检查字段完整性、单位和不变量。
6. Aggregator 仅汇总 schema、suite、环境和 track 一致的数据。
7. Report/Dashboard 从聚合后的正式数据读取，不扫描混杂的旧 `runs/latest`。

关键不变量：

- Text 与 Protocol 任务内容哈希相同。
- Cold 运行开始时记忆库为空。
- Warm 只允许访问对应 Build 产生的记忆。
- 每次 state read 都有访问字节记录。
- 汇总记录可追溯到原始 trace。

## 7. 错误处理

- 指标字段缺失：本轮标记 invalid，不以 0 参与汇总。
- 模型或外部服务失败：记录失败类型，可按预设次数重试，但不得只保留成功样本。
- 记忆 payload 损坏：拒绝复用，回退完整链路。
- SHM 不可用：记录 fallback 原因并使用文件 backend，不得伪报 SHM 次数。
- 质量校验失败：结果计入失败率，不能通过生成更长答案提高分数。
- schema 版本不兼容：Dashboard 显示诊断，不跨版本聚合。

## 8. 测试策略

### 单元测试

- 指标公式和单位。
- tokenizer 计数。
- ReuseDecision 阈值与拒绝条件。
- Warm/Cold 数据隔离。
- StateRef/SHM 字节统计。
- 质量 evaluator。
- schema migration 和旧结果拒绝逻辑。

### 集成测试

- 四 Agent Text/Protocol 配对运行。
- Build/Warm/Cold 完整链路。
- 高置信 verified memory 跳过阶段。
- 低置信或损坏 memory 回退完整链路。
- shared memory 实际创建、读取和清理。
- Rust Core 可用与 Python fallback 行为一致。

### 系统测试

- 离线 10 轮连续任务。
- openEuler Docker 全量质量命令。
- 正式 suite 重复 5 次。
- Dashboard 与报告数值逐项对账。

## 9. openEuler 验收

正式验收环境固定为 openEuler 24.03-LTS-SP3。验收脚本必须输出：

- `/etc/os-release`。
- CPU、内存和架构。
- Python、uv、Rust 和编译器版本。
- Git commit。
- Rust Core 构建结果。
- lint、typecheck、test 结果。
- 离线 benchmark 结果。
- 样例 LLM benchmark 结果。

验收日志写入受版本控制的交付目录，敏感配置和 API key 必须脱敏。

## 10. 推荐执行顺序

| 顺序 | 工作包 | 依赖 | 完成标志 |
|---:|---|---|---|
| 1 | 修复 7 项失败并全量回归 | 无 | 160 项及新增测试全部通过 |
| 2 | 冻结指标 schema 和环境 manifest | 1 | 新旧结果明确隔离 |
| 3 | 重构 Benchmark 配对、重复与统计 | 2 | 10+ 任务、5 repeats、统计量完整 |
| 4 | 重构质量评估 | 3 | 任务断言/测试替代长度评分 |
| 5 | 实现 ReuseDecision 和阶段跳过 | 1、4 | Warm 出现可验证的实际收益 |
| 6 | 完成 StateRef/SHM 消融 | 2 | SHM 真实调用且收益边界清楚 |
| 7 | 重构报告与 Dashboard | 3、4、5、6 | 图表可追溯原始数据 |
| 8 | openEuler 全量验收 | 1 至 7 | 验收脚本和日志齐全 |
| 9 | 正式报告与演示材料 | 7、8 | 交付清单全部完成 |

## 11. 里程碑

### M1：工程可信

- 全部测试通过。
- 指标 schema 稳定。
- 10 轮连续运行通过。

### M2：实验可信

- 公平 Benchmark 上线。
- 主实验完成 5 次重复。
- 质量评价与统计输出有效。

### M3：核心结论成立

- 三套主实验 token 节省为正。
- Warm/Cold 至少一项资源指标改善 20%。
- StateRef/SHM 消融能够说明适用边界。

### M4：可提交

- openEuler 验收通过。
- 固定样例 artifact 已提交。
- 报告、Dashboard、演示脚本和视频分镜一致。

## 12. 完成定义

只有同时满足以下条件，才能宣称冲分版完成：

- 主分支工作区干净，lint、typecheck、test 全绿。
- 正式实验配置、模型/离线 track、seed、repeat 和环境均可追溯。
- 所有主图表可以从提交的原始数据重新生成。
- 不使用回答长度作为质量代理。
- 不使用 memory hit rate 代替 memory benefit。
- 不使用只计算 envelope 的 bytes 代表端到端通信成本。
- 单次连续任务不少于 10 轮。
- openEuler 24.03-LTS-SP3 上完成构建、测试和 benchmark。
- 技术报告中的每个性能结论都有对应原始记录。
- AgentShell 核心命令、裸文本问答、流式输出、代码字符保真和异常后继续交互的
  回归测试全部通过。
