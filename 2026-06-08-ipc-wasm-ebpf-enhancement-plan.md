# AgentMesh-Runtime 系统底层技术增强计划（修订版 v2）

> 面向赛题加分项：IPC / Socket / 共享内存 / 沙箱增强 / eBPF
> 目标系统：openEuler 24.03-LTS-SP3
> 计划制定：2026-06-08 | 修订：v2

---

## 目录

1. [修订说明](#1-修订说明)
2. [现状分析](#2-现状分析)
3. [总体架构](#3-总体架构)
4. [阶段一：Transport 抽象与 Rust Socket Transport](#4-阶段一transport-抽象与-rust-socket-transport)
5. [阶段二：共享内存 State Transport](#5-阶段二共享内存-state-transport)
6. [阶段三：nsjail 沙箱执行引擎（替代 WASM）](#6-阶段三nsjail-沙箱执行引擎替代-wasm)
7. [阶段四：eBPF 可选演示脚本（不接入主流程）](#7-阶段四ebpf-可选演示脚本不接入主流程)
8. [阶段五：集成评测与 Benchmark](#8-阶段五集成评测与-benchmark)
9. [交付清单与工作量评估](#9-交付清单与工作量评估)

---

## 1. 修订说明

### v1 → v2 关键变更

| 条目 | v1 方案 | v2 修订 | 原因 |
|------|--------|---------|------|
| **沙箱** | WASM (wasmtime) | nsjail/seccomp 容器沙箱 | WASM 无法可靠运行 LLM 生成的 Python；Phase A 纯计算示例不覆盖真实 CodeAct |
| **eBPF 定位** | 核心 metrics + BenchmarkSummary | 独立可选演示脚本，不接入主流程 | root/Linux 依赖破坏可复现性；Observer Effect 污染 benchmark 数据 |
| **沙箱工作量** | 4 天（WASM） | 2 天（nsjail） | nsjail 是 C 工具，集成简单 |
| **eBPF 工作量** | 3 天（Rust eBPF + 内核集成） | 1 天（bpftrace 脚本） | 不写 Rust eBPF 代码，纯脚本级 |
| **总工作量** | 14 天 | 10 天 | — |

---

## 2. 现状分析

### 当前架构的局限

```
┌───────────────────────────────────────────────────┐
│                  Agent Runtime                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Planner  │  │Retriever │  │   Executor       │ │
│  └──────────┘  └──────────┘  └──────────────────┘ │
│         │            │               │              │
│         └────────────┴───────────────┘              │
│                        │  AMPMessage in memory      │
│              ┌─────────▼──────────┐                 │
│              │  ProtocolRouter    │                 │
│              │  (同步函数调用)     │                 │
│              └────────────────────┘                 │
│  ┌──────────────────────────────────────────────┐   │
│  │  StateStore  │  MemoryStore  │  SandboxRunner│   │
│  │  (本地文件)   │  (SQLite)     │  (subprocess)  │   │
│  │               │               │  + monkey-patch │   │
│  └──────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────┘
```

关键短板：

| 技术点 | 现状 | 赛题加分方向 |
|--------|------|------------|
| **IPC** | `ProtocolRouter.route()` 是同步函数调用 | 跨进程 Agent 通信，支持分布式部署 |
| **Socket** | Typed envelope 有设计，但无真实网络传输 | Agent 可运行在不同主机，通过 TCP/Unix Socket 通信 |
| **共享内存** | State payload 写文件，通过 `state://` 引用读取 | `state://shm/...` 通过 POSIX 共享内存传递大状态 |
| **沙箱** | `SandboxRunner` 用 `subprocess` + Python monkey-patching 防护 | 强隔离沙箱（nsjail/firejail），对抗 seccomp 绕过 |
| **eBPF** | 无 | eBPF 跟踪通信延迟、内存分配、系统调用（可选演示） |

### 依赖判断

```
    openEuler 24.03-LTS-SP3 原生支持情况
    ├── Unix Domain Socket      ✅ 原生
    ├── POSIX Shared Memory     ✅ (shm_open/mmap)
    ├── TCP Socket              ✅ 原生
    ├── nsjail                  ⚠️ 需编译安装（纯 C，无复杂依赖）
    ├── firejail                ⚠️ dnf install
    ├── eBPF                    ✅ (Linux 内核 ≥ 5.10)
    │   ├── bpftrace            ⚠️ dnf install bpftrace
    │   └── CO-RE BTF           ✅ 默认启用
    └── Rust Core 扩展          ✅ 已前置
```

---

## 3. 总体架构

### 增强后的架构

```
┌─────────────────────────────────────────────────────────────┐
│                Multi-Agent Runtime (Python)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Planner  │  │Retriever │  │ Executor │  │Summarizer  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │              │             │               │         │
│       └──────────────┴─────────────┴───────────────┘         │
│                          │  AMPMessage                        │
│              ┌───────────▼────────────┐                       │
│              │  ProtocolScheduler     │                       │
│              │   (AgentTransport 抽象) │                       │
│              └───┬───────┬───────┬───┘                       │
│                  │       │       │                            │
├──────────────────┼───────┼───────┼────────────────────────────┤
│   ┌──────────────▼──┐┌──▼────┐┌──▼───────────────┐           │
│   │  InProcTransport││ShmState││  SocketTransport  │           │
│   │  (当前, 内存)   ││Store   ││(TCP/Unix Socket)   │           │
│   └─────────────────┘└───────┘└──────────────────┘           │
│   ┌──────────────────────────────────────────────────────┐    │
│   │  Rust Core (agentmesh_core)                          │    │
│   │  codec / embedding / state_ref / vector_index /      │    │
│   │  envelope / sandbox_pool → 可选扩展                   │    │
│   └──────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐   ┌──────────────────┐
│  SandboxRunner 增强                            │   │  eBPF 演示 (可选)  │
│  subprocess (fallback)                        │   │  ┌──────────────┐│
│  → nsjail (主路径，openEuler 编译可用)          │   │  │bpftrace .bt  ││
│  → warm_worker (低延迟路径)                    │   │  │bcc .py       ││
│  → Rust sandbox_pool (可选)                   │   │  │demo_guide.md ││
└──────────────────────────────────────────────┘   │  → 不接入主流程  │
                                                   └──────────────┘
```

### 核心设计原则

1. **Transport 抽象层** — 协议路由不感知传输方式，`AMPMessage` 的发送/接收通过 `AgentTransport` 抽象
2. **渐进替换** — 新增 Transport 不破坏 `InProcTransport`（当前的内存调用），benchmark 可对比
3. **可复现第一** — nsjail 有 Python fallback 路径，eBPF 完全独立于主流程
4. **可评测** — 每种传输/沙箱方式都有 metrics 统计

---

## 4. 阶段一：Transport 抽象与 Rust Socket Transport

### 4.1 Transport 抽象

**文件：** `src/agentmesh/protocol/transport.py`（已在本次优化中完成）

```python
@dataclass
class TransportMetrics:
    transport_type: str
    send_count: int = 0
    total_bytes: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float: ...
    @property
    def p99_latency_ms(self) -> float: ...
    def record(self, *, byte_count: int, latency_ms: float) -> None: ...

class AgentTransport(ABC):
    @property
    @abstractmethod
    def transport_type(self) -> str: ...
    @abstractmethod
    def send(self, message: AMPMessage, timeout=None) -> AMPMessage: ...
    @abstractmethod
    def metrics(self) -> TransportMetrics: ...

class InProcTransport(AgentTransport):
    """进程内直接函数调用，保持当前 benchmark 基线"""
    ...
```

### 4.2 Scheduler 集成（已在本次优化中完成）

`ProtocolScheduler` 通过 `InProcTransport` 调用 agent，而不是直接 `agent.handle()`。`transport_metrics()` 暴露统计数据。

### 4.3 Rust Socket Transport（下一步实现）

**新增文件：** `crates/agentmesh-core/src/socket_transport.rs`

**帧协议（与 Python `SocketFrameTransport` 一致）：**

```text
┌──────────────┬──────────────────────────────┐
│  4 bytes     │    variable                  │
│  payload_len │    msgpack(AMPMessage)       │
└──────────────┴──────────────────────────────┘
```

**设计要点：**

| 连接方式 | 适用场景 | 实现路径 |
|---------|---------|---------|
| `tcp://host:port` | 跨主机 Agent | `TcpStream` + 长度前缀帧 |
| `unix:///tmp/agentmesh.sock` | 同主机跨进程 | `UnixStream` + 长度前缀帧 |
| `inproc://agent-name` | 当前进程内 | 直接函数调用（fallback） |

**连接方式：** Rust `SocketTransport` 实现 `send()` 方法，返回对端 Agent 的 `RESULT`。Python 端包装为 `AgentTransport` 接口。

### 4.4 配置方式

`.env` 新增变量：

```bash
# Transport 类型: inproc | tcp | unix
AGENTMESH_TRANSPORT_TYPE=tcp
AGENTMESH_TRANSPORT_ENDPOINT=127.0.0.1:0
AGENTMESH_TRANSPORT_UNIX_SOCKET=/tmp/agentmesh.sock
```

### 4.5 测试

**新增测试：** `tests/test_socket_transport.py`
1. InProcTransport 消息往返（回归）
2. Unix Socket 本地传输
3. TCP localhost 传输
4. Transport metrics 统计正确性

---

## 5. 阶段二：共享内存 State Transport

### 5.1 问题分析

当前 `StateStore` 传递大状态的方式：

```
Agent A 写入 payload → JSON 文件 → Agent B 从文件读取
```

对于 embedding 向量（384维 float64 ≈ 3KB）、代码执行结果等，文件 I/O 是可避免的瓶颈。
共享内存方案通过 `multiprocessing.shared_memory` 零拷贝传递：

```
Agent A 写入 payload → POSIX shm → Agent B 零拷贝读取
```

### 5.2 Python 共享内存后端（已在本次优化中完成）

在 `StateStore._put()` 中新增阈值策略：

```python
def _put(self, ..., payload) -> str:
    encoded = orjson.dumps(payload)
    if self.payload_backend == "shm" and len(encoded) >= self.shm_threshold_bytes:
        payload_ref = self._write_shm_payload(state_id, encoded)
        # → "shm://agentmesh_{state_id}/{size}"
    else:
        payload_path = self.paths.state_payload_dir / f"{state_id}.json"
        payload_path.write_bytes(encoded)
        payload_ref = str(payload_path)
    return self._record(record)
```

### 5.3 StateRef 格式（已在本次优化中完成）

```text
state://text/{uuid}       → 文件（默认）
state://shm/{name}/{size}  → 共享内存（当 threshold 触发）
```

`get()` 方法根据 `payload_ref.startswith("shm://")` 自动选择读取路径。

### 5.4 Rust 共享内存后端（可选加速）

如果 Rust Core 已安装，可以将 shm 操作下沉到 Rust，获得更底层的控制：

**新增文件：** `crates/agentmesh-core/src/shm_state.rs`

```rust
use libc::{shm_open, mmap, shm_unlink, MAP_SHARED, PROT_READ, PROT_WRITE};

// 通过 Rust 的 shm_open/mmap 操作，减少 Python-GIL 开销
pub fn shm_write(name: &str, data: &[u8]) -> PyResult<usize>;
pub fn shm_read(name: &str, size: usize) -> PyResult<Vec<u8>>;
pub fn shm_destroy(name: &str) -> PyResult<()>;
```

Python 端回退逻辑：Rust 不可用时使用 `multiprocessing.shared_memory`（已在 v1 实现）。

### 5.5 指标

- `state_shm_transfer_count` — shm 传递次数
- `state_shm_transfer_bytes` — shm 传递总字节数
- 埋点在 `StateStore.shm_transfer_count()` 和 `shm_transfer_bytes()`（已在本次优化中完成）

### 5.6 测试

**新增测试：** `tests/test_state_store.py`（已在本次优化中完成）
1. 大 payload 自动走 shm
2. 小 payload 走文件降级
3. 跨 trace shm 统计
4. `close()` 后资源释放

---

## 6. 阶段三：nsjail 沙箱执行引擎（替代 WASM）

### 6.1 为什么替换 WASM

| 问题 | 说明 |
|------|------|
| **没有 Python→WASM 编译器** | pyodide 是把整个 CPython 解释器编译到 WASM（>100MB），不能用；市面上没有工具能把任意 Python 代码编译为 `.wasm` |
| **LLM 生成的代码不可控** | CodeAct 场景下，LLM 可能生成 `import os`、`open()`、`subprocess.run()`。Phase A 的"纯计算 subset"过滤器要么拦太多要么漏放 |
| **wasmtime 编译慢** | openEuler 上从源码编译 wasmtime 耗时 ~半天 |
| **答辩风险** | "评委问 LLM 生成的 os.system 怎么拦截"——WASM 回答不了 |

### 6.2 为什么选择 nsjail

| 工具 | 隔离原理 | 启动延迟 | openEuler 可用 | 集成复杂度 |
|------|---------|:--------:|:-------------:|:---------:|
| **nsjail** (Google) | seccomp-bpf + namespace | **1-3ms** | ✅ 源码编译 | 低 |
| firejail | setuid + seccomp | ~10ms | ✅ `dnf install` | 低 |
| Docker/podman | 完整容器 | ~500ms | ✅ podman 内置 | 中 |
| WASM (原方案) | — | — | ❌ 不适用 | 高 |

- nsjail 是 Google 开源工具，CTF/LeetCode Judge 都在用
- 基于 seccomp-bpf 系统调用过滤，比 Python monkey-patch 强几个数量级
- 启动延迟 1-3ms，远低于 docker
- 可精确限制 CPU/memory/time/pid/网络/文件系统
- openEuler 上编译安装：纯 C，依赖少

### 6.3 实现方案

**修改：** `src/agentmesh/sandbox/runner.py`

```python
class NsJailSandboxRunner(SandboxRunner):
    """通过 nsjail 执行 Python 代码，提供 seccomp-bpf 级别隔离"""

    def __init__(self, base_dir, *, fallback=None):
        self._nsjail_available = self._check_nsjail()
        self._fallback = fallback or SubprocessSandboxRunner(base_dir)

    def run_python(self, code: str) -> SandboxResult:
        if not self._nsjail_available:
            return self._fallback.run_python(code)  # 降级

        script = self._write_script(code)
        cmd = [
            "nsjail",
            "--chroot", "/",
            "--rw", str(self.sandbox_dir),          # 只允许写入沙箱目录
            "--bindmount_ro", "/usr",                # 只读系统
            "--bindmount_ro", sys.executable,
            "--disable_clone_newnet",                # 禁止网络
            "--rlimit_as", "256",                    # 256 MB 内存上限
            "--rlimit_cpu", "10",                    # 10 CPU 秒
            "--rlimit_nproc", "10",                  # 最多 10 进程
            "--time_limit", "30",                    # 30 秒超时
            "--", sys.executable, str(script),
        ]
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SandboxResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            latency_ms=latency_ms,
            backend="nsjail",
        )

    def _check_nsjail(self) -> bool:
        try:
            subprocess.run(["nsjail", "--version"], capture_output=True, timeout=2)
            return True
        except FileNotFoundError:
            return False
```

### 6.4 与现有 SandboxRunner 集成

```python
class SandboxRunner:
    def __init__(self, base_dir, *, backend="auto"):
        self._nsjail = NsJailSandboxRunner(base_dir)
        self._subprocess = SubprocessSandboxRunner(base_dir)
        self._warm = WarmWorkerSandboxRunner(base_dir)
        self._backend = self._select_backend(backend)

    def run_python(self, code: str) -> SandboxResult:
        return self._backend.run_python(code)

    def _select_backend(self, backend: str):
        if backend == "nsjail" and self._nsjail._nsjail_available:
            return self._nsjail
        if backend == "warm":
            return self._warm
        if backend == "rust" and rust_available():
            return self._rust_backend()
        return self._subprocess  # 默认 fallback
```

### 6.5 各沙箱后端对比

| 维度 | subprocess (当前) | warm_worker | nsjail | Rust sandbox_pool |
|------|:----------------:|:-----------:|:-----:|:-----------------:|
| 隔离强度 | ❌ monkey-patch | ❌ monkey-patch | ✅ seccomp+namespace | ❌ subprocess |
| 启动延迟 | ~50ms | ~1ms | ~1-3ms | ~50ms |
| 内存限制 | 无 | 无 | ✅ rlimit_as | 无 |
| 网络限制 | 部分（patch） | 部分（patch） | ✅ CLONE_NEWNET | 部分（patch）|
| 超时控制 | ✅ subprocess timeout | ✅ 轮询 | ✅ --time_limit | ✅ 轮询 |
| 安装依赖 | 无 | 无 | ⚠️ 需编译 | ✅ Rust Core |

### 6.6 安装脚本

**新增：** `scripts/setup_nsjail_openeuler.sh`

```bash
#!/usr/bin/env bash
# openEuler 24.03-LTS-SP3 上编译安装 nsjail
set -euo pipefail

sudo dnf install -y gcc gcc-c++ make git flex bison

git clone https://github.com/google/nsjail.git /tmp/nsjail
cd /tmp/nsjail
make
sudo cp nsjail /usr/local/bin/
nsjail --version  # 验证安装
```

### 6.7 测试

**新增测试：** `tests/test_nsjail_sandbox.py`
1. nsjail 可用时正确执行 Python
2. 网络访问被拒绝（socket 调用返回错误）
3. 内存超限被 kill
4. nsjail 不可用时降级到 subprocess
5. 隔离度对比测试（monkey-patch vs nsjail）

---

## 7. 阶段四：eBPF 可选演示脚本（不接入主流程）

### 7.1 设计原则

**不做的事情：**
- ❌ 不往 `RunMetrics` 加 eBPF 字段
- ❌ 不往 `BenchmarkSummary` 加 eBPF 字段
- ❌ 不在 `protocol_mode.py` 中导入或调用 eBPF 代码
- ❌ 不在 benchmark pipeline 中增加任何 eBPF 依赖

**做的事情：**
- ✅ 一组独立的 bpftrace 单行脚本和 bcc Python 脚本
- ✅ 一份答辩演示指南
- ✅ 一个 CLI `agentmesh trace ebpf`（仅在 Linux+root 可用，打印友好提示）

### 7.2 新增脚本

```
scripts/ebpf/
├── trace_socket.bt        # bpftrace：追踪 Socket 通信延迟
├── trace_syscall.py       # bcc Python：系统调用频率统计
├── demo_guide.md          # 答辩演示说明
└── README.md              # 环境要求和运行方式
```

### 7.3 使用方式

```bash
# 方式一：bpftrace 单行脚本
sudo bpftrace scripts/ebpf/trace_socket.bt $(pgrep -f agentmesh)

# 方式二：benchmark 完成后附加分析
# 先开一个终端跑 trace
sudo python scripts/ebpf/trace_syscall.py --pid WAIT --duration 60
# 再开另一个终端跑 benchmark
uv run agentmesh benchmark --suite examples/benchmarks/continuous_tasks.yaml
# eBPF 输出 Socket 调用次数、平均延迟、上下文切换等
```

### 7.4 答辩演示价值

在答辩时展示（可选环节）：

> "我们还在 openEuler 上使用 eBPF 对通信延迟做了内核级观测。这是 Socket Transport 场景下 sendto 系统调用的延迟分布直方图，可以看到 99% 的调用在 50µs 以内完成。这些观测是可选演示环节，不影响 benchmark 的可复现性。"

演示数据示例：

```
@send_latency_hist (ns):
[1K, 4K)      ████████████████  42
[4K, 16K)     ██████             18
[16K, 64K)    ███                 8
[64K, 256K)   ▏                    2

send_count:  70
send_bytes:  284,032
context_switches: 1,203
page_faults: 7
```

### 7.5 不接入主流程的理由

| 问题 | 严重程度 | 说明 |
|------|:-------:|------|
| 不可复现 | 🔴 | root + Linux 5.10+ + BTF 是硬依赖，macOS/Windows/CI 直接失效 |
| 评审重点偏移 | 🟡 | 评委看的是"结构化通信 vs 文本通信"对比，不是内核追踪图 |
| Observer Effect | 🟡 | eBPF 探针加载本身有开销，可能污染延迟数据 |
| 复杂度收益比低 | 🟢 | 用 50 行 bpftrace 脚本能达到同样展示效果，不需要 Rust 集成 |

---

## 8. 阶段五：集成评测与 Benchmark

### 8.1 Transport 对比评测

**新增：** `examples/benchmarks/transport_compare.yaml`

```yaml
name: transport_compare
repeat: 3
tasks:
  - id: A1
    input_file: "examples/tasks/A1_requirements.txt"
  - id: A2
    input_file: "examples/tasks/A2_architecture.txt"
```

运行方式：

```bash
# InProc baseline（默认）
uv run agentmesh benchmark --suite examples/benchmarks/transport_compare.yaml

# 指定 Transport
AGENTMESH_TRANSPORT_TYPE=unix uv run agentmesh benchmark ...
AGENTMESH_TRANSPORT_TYPE=tcp  uv run agentmesh benchmark ...
```

预期输出指标：

```text
Transport Type     SendCnt    Bytes     AvgLat    P99Lat
─────────────────────────────────────────────────────────
InProc                42      12KB      0.02ms    0.05ms
Unix Socket           42      12KB      0.12ms    0.35ms
TCP Loopback          42      12KB      0.28ms    0.72ms
```

### 8.2 沙箱后端对比评测

**新增：** `examples/benchmarks/sandbox_compare.yaml`

```yaml
name: sandbox_compare
repeat: 5
scenarios:
  - backend: subprocess
    label: "Python subprocess"
  - backend: warm_worker
    label: "Warm Python worker"
  - backend: nsjail
    label: "nsjail sandbox"
```

预期输出指标：

```text
Sandbox Backend     Startup    PeakMem    Throughput    Isolation
─────────────────────────────────────────────────────────────────
subprocess           52ms      64MB       19/s          ❌ weak
warm_worker           1ms      68MB       950/s         ❌ weak
nsjail                2ms      16MB       450/s         ✅ seccomp
```

### 8.3 Benchmark 汇总字段

已在 `BenchmarkSummary` 中新增（本次优化已完成）：

```python
class BenchmarkSummary(BaseModel):
    # ... 原有字段 ...
    transport_send_count: int
    transport_bytes: int
    transport_avg_latency_ms: float
    transport_p99_latency_ms: float
    state_shm_transfer_count: int
    state_shm_transfer_bytes: int
```

### 8.4 eBPF 演示报告

eBPF 数据不进入 `BenchmarkSummary`，而是由 `scripts/ebpf/demo_guide.md` 描述如何独立运行和展示。

---

## 9. 交付清单与工作量评估

### 9.1 文件变更清单

| 编号 | 文件 | 操作 | 预估行数 | 阶段 | 状态 |
|------|------|------|---------|:----:|:----:|
| 1 | `src/agentmesh/protocol/transport.py` | **新增** | 130 | P1 | ✅ 已完成 |
| 2 | `src/agentmesh/runtime/scheduler.py` | **修改** | +8 | P1 | ✅ 已完成 |
| 3 | `tests/test_transport.py` | **新增** | 60 | P1 | ✅ 已完成 |
| 4 | `crates/agentmesh-core/src/socket_transport.rs` | **新增** | 350 | P1 | ⬜ 待实现 |
| 5 | `crates/agentmesh-core/src/lib.rs` | **修改** | +10 | P1 | ⬜ 待实现 |
| 6 | `.env.example` | **修改** | +10 | P1 | ✅ 已完成 |
| 7 | `src/agentmesh/config.py` | **修改** | +19 | P2 | ✅ 已完成 |
| 8 | `src/agentmesh/state/store.py` | **修改** | +99 | P2 | ✅ 已完成 |
| 9 | `src/agentmesh/eval/metrics.py` | **修改** | +7 | P2 | ✅ 已完成 |
| 10 | `src/agentmesh/eval/benchmark.py` | **修改** | +34 | P2 | ✅ 已完成 |
| 11 | `src/agentmesh/eval/report.py` | **修改** | +6 | P2 | ✅ 已完成 |
| 12 | `src/agentmesh/modes/protocol_mode.py` | **修改** | +15 | P2 | ✅ 已完成 |
| 13 | `tests/test_config.py` | **修改** | +19 | P2 | ✅ 已完成 |
| 14 | `tests/test_state_store.py` | **修改** | +45 | P2 | ✅ 已完成 |
| 15 | `tests/test_modes_and_benchmark.py` | **修改** | +34 | P2 | ✅ 已完成 |
| 16 | `src/agentmesh/sandbox/runner.py` | **修改** | +120 | P3 | ⬜ 待实现 |
| 17 | `scripts/setup_nsjail_openeuler.sh` | **新增** | 20 | P3 | ⬜ 待实现 |
| 18 | `tests/test_nsjail_sandbox.py` | **新增** | 150 | P3 | ⬜ 待实现 |
| 19 | `scripts/ebpf/trace_socket.bt` | **新增** | 40 | P4 | ⬜ 待实现 |
| 20 | `scripts/ebpf/trace_syscall.py` | **新增** | 80 | P4 | ⬜ 待实现 |
| 21 | `scripts/ebpf/demo_guide.md` | **新增** | 50 | P4 | ⬜ 待实现 |
| 22 | `examples/benchmarks/transport_compare.yaml` | **新增** | 20 | P5 | ⬜ 待实现 |
| 23 | `examples/benchmarks/sandbox_compare.yaml` | **新增** | 25 | P5 | ⬜ 待实现 |
| 24 | `docs/superpowers/2026-06-08-ipc-wasm-ebpf-enhancement-plan.md` | **修订** | 本文 | — | ✅ 进行中 |

### 9.2 新增依赖

```toml
# crates/agentmesh-core/Cargo.toml 增量
[dependencies]
# 已有
pyo3 = { version = "0.22", features = ["extension-module"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rmp-serde = "1"
sha2 = "0.10"

# P1 新增
# (std::net + std::os::unix 无需额外依赖)

# P2 Rust shm 后端（可选，无依赖时走 Python multiprocessing.shared_memory）
libc = "0.2"    # 仅 Rust shm_open/mmap 需要
```

nsjail 为独立 C 工具，不进入 Python/Rust 依赖。

### 9.3 工作量评估

| 阶段 | 内容 | 预估人天 | 前置依赖 | 状态 |
|:----:|------|:-------:|---------|:----:|
| **P1** | Transport 抽象 + Socket Transport | 2 天 | Rust Core | 🟡 Python 层已完成，Rust 层待实现 |
| **P2** | 共享内存 State Transport | 1 天 | 无 | ✅ 已完成 |
| **P3** | nsjail 沙箱 | 2 天 | 无 | ⬜ 待实现 |
| **P4** | eBPF 可选演示 | 1 天 | openEuler Linux | ⬜ 待实现 |
| **P5** | 集成评测 | 1 天 | P1-P3 | 🟡 字段已加，YAML 待补 |
| **合计** | | **7 天** | | |

**v1 → v2 工作量变化：** 14 天 → 7 天，减半。原因是砍掉了 wasmtime 编译调试（~3 天）和 Rust eBPF 集成（~2 天），降低了近一半。

### 9.4 风险与缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|:----:|:----:|---------|
| nsjail 在 openEuler 某内核版本上 seccomp 不兼容 | 低 | 中 | 编译最新版；fallback 到 subprocess |
| nsjail 未安装时 benchmark 不报错 | — | — | `_check_nsjail()` 静默降级，metrics 中标记 backend=subprocess |
| Socket Transport 在 Rust 中实现复杂 | 中 | 中 | Python `SocketFrameTransport` 已可作为纯 Python 替代方案 |
| 共享内存跨进程权限问题 | 低 | 低 | fallback 到文件模式 |
| eBPF 脚本在非 Linux 环境报错 | — | 低 | 脚本独立，与主流程完全解耦，报错不影响核心功能 |

### 9.5 赛题评分点映射

| 赛题评分项 | 分值 | 对应增强 | 预期提升 |
|-----------|:---:|---------|:--------:|
| **通信效率** | 25 | Socket 批量传输、Typed Envelope 网络优化、零拷贝 shm | 在 InProc 基础上展示真实网络环境下的节省 |
| **状态传递创新** | 20 | 共享内存 State Transfer、跨进程 Agent 架构 | 展示跨进程/跨主机非文本状态传递 |
| **系统完整性** | 20 | nsjail 强隔离沙箱 + 分布式部署支持 | 体现系统级技术深度 |
| **实验验证** | 15 | Transport 对比 + Sandbox 对比 benchmark | 多维度、多后端的对比数据 |

### 9.6 答辩展示规划

| 展示内容 | 对应的加分项 | 展示方式 |
|---------|------------|---------|
| Transport 对比表格（InProc vs Unix vs TCP） | 通信效率 | `agentmesh benchmark` CLI 输出 |
| 共享内存 State 传递 | 状态传递创新 | `agentmesh memory stats` + state_index.sqlite 记录 |
| nsjail 沙箱隔离验证 | 系统完整性 | 现场 `nsjail --version` + 防火墙测试 |
| eBPF 延迟直方图（可选） | 技术深度 | 独立脚本演示，不算入 benchmark |

---

## 附录：关键代码示例

### A. Socket Transport 帧协议

```rust
// Rust: 发送带长度前缀的 msgpack 帧
pub fn socket_send(stream: &mut TcpStream, message: &AMPMessage) -> PyResult<()> {
    let payload = rmp_serde::to_vec_named(&message)?;
    let len = payload.len() as u32;
    let header = len.to_be_bytes();
    stream.write_all(&header)?;
    stream.write_all(&payload)?;
    Ok(())
}

pub fn socket_recv(stream: &mut TcpStream) -> PyResult<AMPMessage> {
    let mut header = [0u8; 4];
    stream.read_exact(&mut header)?;
    let len = u32::from_be_bytes(header) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    let message: AMPMessage = rmp_serde::from_slice(&payload)?;
    Ok(message)
}
```

### B. 共享内存 State Transfer

```python
# Python 实现（已完成，使用 multiprocessing.shared_memory）
def _write_shm_payload(self, state_id: str, encoded: bytes) -> str:
    shm = shared_memory.SharedMemory(
        name=f"agentmesh_{state_id.replace('-', '_')}",
        create=True, size=len(encoded),
    )
    shm.buf[:len(encoded)] = encoded
    self._owned_shm[shm.name] = shm
    return f"shm://{shm.name}/{len(encoded)}"
```

```rust
// Rust 可选加速（使用 libc shm_open + mmap）
pub fn shm_write(name: &str, data: &[u8]) -> PyResult<usize> {
    let cname = CString::new(name)?;
    let fd = shm_open(cname.as_ptr(), O_CREAT | O_RDWR, 0o600);
    ftruncate(fd, data.len() as off_t)?;
    let ptr = mmap(ptr::null_mut(), data.len(),
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    unsafe { ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len()) };
    munmap(ptr, data.len()); close(fd);
    Ok(data.len())
}
```

### C. eBPF bpftrace Socket Trace

```c
// scripts/ebpf/trace_socket.bt — 答辩演示用，不接入主流程
tracepoint:syscalls:sys_enter_sendto
/pid == $1/
{
    @send_count++;
    @send_bytes += args->len;
    @send_start[args->fd] = nsecs;
}

tracepoint:syscalls:sys_exit_sendto
/pid == $1/
{
    $start = @send_start[args->fd];
    if ($start > 0) {
        @send_latency_hist = hist(nsecs - $start);
        delete(@send_start[args->fd]);
    }
}

END {
    printf("Socket send count: %d\n", @send_count);
    printf("Socket send bytes: %d\n", @send_bytes);
    printf("Socket send latency (ns):\n");
    print(@send_latency_hist);
}
```
