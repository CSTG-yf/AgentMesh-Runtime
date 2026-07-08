# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import yaml

SUMMARY_FILE = "benchmark_summary.csv"
DETAIL_FILE = "benchmark_detail.jsonl"
REPORT_FILE = "experiment_report.md"


def load_dashboard_data(root: Path) -> dict[str, object]:
    configured = _configured_suites(root)
    generated = _generated_suite_dirs(root)
    names = sorted(configured.keys() | generated.keys())
    suites: list[dict[str, object]] = []
    available_sources = 0
    expected_sources = len(names) * 3

    for name in names:
        config = configured.get(name, {})
        suite_dir = generated.get(name)
        diagnostics: list[dict[str, str]] = []
        summary: dict[str, object] = {}
        details: list[dict[str, object]] = []
        report: str | None = None
        source_count = 0

        if suite_dir is not None:
            summary_path = suite_dir / SUMMARY_FILE
            detail_path = suite_dir / DETAIL_FILE
            report_path = suite_dir / REPORT_FILE
            if summary_path.exists():
                source_count += 1
                summary = _read_summary(summary_path, diagnostics)
            else:
                diagnostics.append(_missing_file(SUMMARY_FILE))
            if detail_path.exists():
                source_count += 1
                details = _read_jsonl(detail_path, diagnostics)
            else:
                diagnostics.append(_missing_file(DETAIL_FILE))
            if report_path.exists():
                source_count += 1
                report = _read_text(report_path, diagnostics)
            else:
                diagnostics.append(_missing_file(REPORT_FILE))

        available_sources += source_count
        status = (
            "not_generated"
            if suite_dir is None or source_count == 0
            else "complete"
            if source_count == 3 and not diagnostics
            else "partial"
        )
        suites.append(
            {
                "name": name,
                "status": status,
                "configured_task_count": _task_count(config),
                "configured_tasks": config.get("tasks", []),
                "summary": summary,
                "details": details,
                "report": report,
                "diagnostics": diagnostics,
                "source_count": source_count,
            }
        )

    completeness = available_sources / expected_sources if expected_sources else 0.0
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "root": str(root.resolve()),
        "completeness": completeness,
        "available_sources": available_sources,
        "expected_sources": expected_sources,
        "suites": suites,
    }


def _task_count(config: dict[str, object]) -> int:
    tasks = config.get("tasks")
    return len(tasks) if isinstance(tasks, list) else 0


def generate_dashboard(root: Path, output: Path | None = None) -> Path:
    data = load_dashboard_data(root)
    destination = output or root / "runs/latest/benchmarks/benchmark_dashboard.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_html(data), encoding="utf-8")
    return destination


def _configured_suites(root: Path) -> dict[str, dict[str, object]]:
    suites: dict[str, dict[str, object]] = {}
    config_dir = root / "examples" / "benchmarks"
    if not config_dir.exists():
        return suites
    for path in sorted((*config_dir.glob("*.yaml"), *config_dir.glob("*.yml"))):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or path.stem)
        tasks = value.get("tasks")
        safe_tasks = []
        if isinstance(tasks, list):
            safe_tasks = [task for task in tasks if isinstance(task, dict)]
        suites[name] = {"tasks": safe_tasks, "config_path": str(path)}
    return suites


def _generated_suite_dirs(root: Path) -> dict[str, Path]:
    benchmark_dir = root / "runs" / "latest" / "benchmarks"
    if not benchmark_dir.exists():
        return {}
    return {
        path.name: path
        for path in benchmark_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def _read_summary(
    path: Path, diagnostics: list[dict[str, str]]
) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            row = next(csv.DictReader(file), None)
    except (OSError, UnicodeError, csv.Error) as exc:
        diagnostics.append(
            {"kind": "malformed_csv", "source": path.name, "message": str(exc)}
        )
        return {}
    if row is None:
        diagnostics.append(
            {"kind": "empty_file", "source": path.name, "message": "No summary row"}
        )
        return {}
    return {str(key): _parse_scalar(value) for key, value in row.items() if key is not None}


def _read_jsonl(
    path: Path, diagnostics: list[dict[str, str]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        diagnostics.append(
            {"kind": "unreadable_file", "source": path.name, "message": str(exc)}
        )
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            diagnostics.append(
                {
                    "kind": "malformed_jsonl",
                    "source": f"{path.name}:{line_number}",
                    "message": exc.msg,
                }
            )
            continue
        if not isinstance(value, dict):
            diagnostics.append(
                {
                    "kind": "malformed_jsonl",
                    "source": f"{path.name}:{line_number}",
                    "message": "Row is not a JSON object",
                }
            )
            continue
        rows.append(value)
    return rows


def _read_text(path: Path, diagnostics: list[dict[str, str]]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        diagnostics.append(
            {"kind": "unreadable_file", "source": path.name, "message": str(exc)}
        )
        return None


def _parse_scalar(value: str | None) -> object:
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    try:
        number = float(normalized)
    except ValueError:
        return normalized
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() and "." not in normalized else number


def _missing_file(name: str) -> dict[str, str]:
    return {"kind": "missing_file", "source": name, "message": "Artifact not generated"}


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _render_html(data: dict[str, object]) -> str:
    return _HTML.replace("__DASHBOARD_DATA__", _safe_json(data))


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentMesh 基准测试（Benchmark）</title>
<style>
:root{color-scheme:dark;--bg:#061523;--bg2:#081a2a;--panel:#0a1e2f;--panel2:#0c2337;--line:#29445a;--line2:#1c3549;--text:#f1f5f9;--muted:#9db0c0;--blue:#2f81f7;--blue2:#60a5fa;--slate:#98a8b6;--green:#49c887;--amber:#f4b860;--red:#f87171;--radius:8px}
*{box-sizing:border-box}
html{background:var(--bg);scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
button,input,select{font:inherit}
button{color:inherit}
.shell{max-width:1600px;margin:auto;padding:22px}
.topbar{display:flex;align-items:flex-start;justify-content:space-between;gap:28px;margin-bottom:14px}
h1{margin:0;font-size:32px;line-height:1.1;letter-spacing:-.035em;font-weight:720}
.meta{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:10px 18px;color:var(--muted);padding-top:8px}
.meta span+span{padding-left:18px;border-left:1px solid var(--line)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--slate);margin-right:7px}
.workspace{display:grid;grid-template-columns:220px minmax(0,1fr);gap:12px;align-items:start}
.rail,.main-panel,.kpis,.section{border:1px solid var(--line);background:linear-gradient(180deg,rgba(10,30,47,.96),rgba(6,21,35,.98));border-radius:var(--radius)}
.rail{position:sticky;top:12px;min-height:calc(100vh - 72px);padding:10px;display:flex;flex-direction:column}
.rail-title,.section-title{font-size:12px;text-transform:uppercase;letter-spacing:.075em;font-weight:700;color:#c7d3dc}
.rail-title{padding:7px 8px 12px}
.suite-list{display:grid;gap:5px}
.suite{display:block;width:100%;text-align:left;background:transparent;border:1px solid transparent;border-radius:6px;padding:11px 10px;cursor:pointer}
.suite:hover{background:#0d263b}.suite.active{border-color:var(--blue);background:#0d2944;box-shadow:inset 3px 0 var(--blue)}
.suite-name{display:block;font-weight:650;overflow-wrap:anywhere}.suite-meta{display:block;color:var(--muted);margin-top:5px}
.suite-status{display:block;margin-top:7px;color:var(--muted)}.suite-status .dot{background:var(--green)}.suite-status.partial .dot{background:var(--amber)}.suite-status.not_generated .dot{background:var(--slate)}
.legend{margin-top:auto;border-top:1px solid var(--line2);padding:14px 8px 4px;color:var(--muted);display:grid;gap:7px}.legend .blue{background:var(--blue)}.legend .green{background:var(--green)}.legend .amber{background:var(--amber)}
.content{min-width:0;display:grid;gap:10px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);padding:14px 0}
.kpi{padding:3px 24px;min-width:0}.kpi+.kpi{border-left:1px solid var(--line)}
.kpi-label{color:#b8c7d3;font-size:12px}.kpi-value{font-size:30px;line-height:1.15;font-weight:720;margin-top:8px;letter-spacing:-.03em}
.kpi-note{color:var(--muted);margin-top:3px}.na{color:#7f95a6}
.section{overflow:hidden}
.section-head{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:10px 14px;border-bottom:1px solid var(--line2)}
.section-head small{color:var(--muted);font-size:11px}
.mode-key{display:flex;gap:18px;color:#c9d4dc}.key-box{display:inline-block;width:11px;height:11px;margin-right:7px;background:var(--slate)}.key-box.protocol{background:var(--blue)}
.charts{display:grid;grid-template-columns:repeat(4,1fr);padding:14px}
.chart{padding:0 16px;min-width:0}.chart+.chart{border-left:1px solid var(--line)}
.chart h3{font-size:12px;margin:0 0 14px}.bar-row{display:grid;grid-template-columns:74px minmax(30px,1fr);gap:8px;align-items:center;margin:10px 0;color:#b9c6cf}
.track{height:12px;background:#102a3f;position:relative}.bar{height:100%;background:var(--slate);min-width:0}.bar.protocol{background:var(--blue)}
.bar-value{margin-top:5px;text-align:right;color:var(--muted);font-size:11px}.chart-empty{height:72px;display:grid;place-items:center;color:var(--muted);border:1px dashed var(--line)}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.field{height:32px;border:1px solid var(--line);border-radius:5px;background:#081a29;color:var(--text);padding:0 10px}.search{width:260px}.clear{height:32px;border:1px solid var(--line);background:#112b40;border-radius:5px;padding:0 13px;cursor:pointer}
.table-wrap{overflow:auto}.results{width:100%;border-collapse:collapse;white-space:nowrap}.results th,.results td{padding:8px 11px;border-right:1px solid var(--line);border-bottom:1px solid var(--line2);text-align:right}.results th:first-child,.results td:first-child,.results th:nth-child(2),.results td:nth-child(2),.results th:nth-child(3),.results td:nth-child(3){text-align:left}
.results th{position:sticky;top:0;background:#102a3f;color:#d8e1e8;font-size:11px;letter-spacing:.02em;cursor:pointer}.results td{color:#c7d3dc}.results tr:hover td{background:#0c2538}.mode-protocol{color:var(--blue2)!important}
.empty{padding:36px;text-align:center;color:var(--muted)}.empty strong{display:block;color:#ced8df;font-size:14px;margin-bottom:4px}
.metrics{display:grid;grid-template-columns:repeat(5,1fr)}.metric-group{padding:14px;border-right:1px solid var(--line);min-width:0}.metric-group:last-child{border:0}.metric-group h3{margin:0 0 10px;font-size:11px;text-transform:uppercase;letter-spacing:.07em}.metric-row{display:flex;gap:10px;justify-content:space-between;margin:6px 0;color:#b9c7d1}.metric-row span:first-child{overflow-wrap:anywhere}.metric-row code{color:#e4ebf0;font:inherit;text-align:right}.metric-row .na{font-style:italic}
details summary{cursor:pointer;list-style:none;padding:11px 14px;font-weight:650;text-transform:uppercase;letter-spacing:.06em;font-size:11px}details summary::-webkit-details-marker{display:none}.evidence{border-top:1px solid var(--line2);display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px}.evidence pre{margin:0;max-height:300px;overflow:auto;background:#06131f;border:1px solid var(--line2);padding:12px;color:#aebdca;white-space:pre-wrap;overflow-wrap:anywhere}.diagnostics{display:grid;gap:7px}.diagnostic{border-left:2px solid var(--amber);padding:8px 10px;background:#102438}.diagnostic strong{display:block;color:#e8d3ad}
.nosuites{padding:45px;text-align:center}.nosuites h2{margin:0 0 8px}.nosuites p{color:var(--muted)}
.footer{color:#73899a;font-size:11px;text-align:right;padding:7px 2px}
@media(max-width:1100px){.workspace{grid-template-columns:1fr}.rail{position:static;min-height:0}.suite-list{grid-template-columns:repeat(3,1fr)}.legend{display:none}.charts{grid-template-columns:1fr 1fr}.chart:nth-child(3){border-left:0}.chart:nth-child(n+3){border-top:1px solid var(--line);padding-top:14px}.metrics{grid-template-columns:1fr 1fr 1fr}.metric-group{border-bottom:1px solid var(--line)}}
@media(max-width:720px){.shell{padding:12px}.topbar{display:block}.meta{justify-content:flex-start}.meta span+span{border:0;padding-left:0}.suite-list{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.kpi{padding:12px 16px}.kpi+.kpi{border-left:0}.charts,.metrics,.evidence{grid-template-columns:1fr}.chart+.chart,.metric-group{border-left:0;border-right:0;border-top:1px solid var(--line);padding-top:14px}.section-head{align-items:flex-start;flex-direction:column}.toolbar{width:100%}.search{width:100%}.kpi-value{font-size:25px}}
</style>
</head>
<body>
<main class="shell">
  <header class="topbar">
    <h1>AgentMesh 基准测试（Benchmark）</h1>
    <div class="meta">
      <span id="generated"></span>
      <span><i class="dot"></i>离线报告（Offline report）</span>
      <span id="completeness"></span>
    </div>
  </header>
  <div id="app"></div>
  <div class="footer">独立报告 · 无需外部运行时依赖</div>
</main>
<script>
const DATA=__DASHBOARD_DATA__;
const $=(q,e=document)=>e.querySelector(q);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const number=v=>finite(v)?new Intl.NumberFormat("en-US",{maximumFractionDigits:2}).format(v):"N/A";
const pct=v=>finite(v)?`${(v*100).toFixed(1)}%`:"N/A";
const human=v=>{if(!finite(v))return"N/A";for(const [n,d] of [["GB",1e9],["MB",1e6],["KB",1e3]])if(Math.abs(v)>=d)return`${(v/d).toFixed(2)} ${n}`;return number(v)};
function zhLabel(k){
 const map={
  total_runs:"总运行次数",token_estimator:"Token 估算器",token_saving_rate:"Token 节省率",
  agent_io_bytes_reduction_rate:"Agent I/O 字节减少率",
  text_agent_io_tokens:"文本模式 Agent I/O Token",protocol_agent_io_tokens:"协议模式 Agent I/O Token",
  text_per_msg_avg_tokens:"文本模式平均每消息 Token",protocol_per_msg_avg_tokens:"协议模式平均每消息 Token",
  latency_reduction_rate:"延迟降低率",quality_preservation_rate:"质量保持率",
  wire_bytes_reduction_rate:"Wire 字节减少率",fair_wire_reduction_rate:"公平 Wire 减少率",
  text_agent_io_bytes:"文本模式 Agent I/O 字节",protocol_agent_io_bytes:"协议模式 Agent I/O 字节",
  protocol_total_bytes:"协议总字节",text_wire_bytes:"文本模式 Wire 字节",
  protocol_wire_bytes:"协议模式 Wire 字节",
  protocol_session_dictionary_bytes:"协议会话字典（Session Dictionary）字节",
  protocol_typed_envelope_bytes:"协议类型化信封（Typed Envelope）字节",
  protocol_typed_payload_bytes:"协议类型化负载（Typed Payload）字节",
  protocol_compact_message_bytes:"协议压缩消息（Compact Message）字节",
  protocol_json_wire_bytes:"协议 JSON Wire 字节",
  protocol_state_payload_bytes:"协议状态负载（State Payload）字节",
  memory_hit_rate:"记忆命中率",memory_reused_unit_count:"记忆重用单元数",
  memory_evidence_count:"记忆证据数",memory_evidence_bytes:"记忆证据字节",
  memory_avg_evidence_bytes_per_query:"每次查询平均证据字节",
  memory_avg_reused_units_per_query:"每次查询平均重用单元数",
  memory_avg_score:"平均记忆评分",memory_avg_semantic_similarity:"平均语义相似度（Semantic Similarity）",
  memory_avg_tag_overlap_score:"平均标签重叠评分",
  rust_core_enabled_runs:"Rust Core 启用运行数",rust_sandbox_backend_runs:"Rust Sandbox 后端运行数",
  transport_send_count:"传输发送次数",transport_bytes:"传输字节数",
  transport_avg_latency_ms:"平均传输延迟",transport_p99_latency_ms:"传输延迟 P99",
  state_shm_transfer_count:"状态 SHM 传输次数",state_shm_transfer_bytes:"状态 SHM 传输字节",
  feedback_round_count:"反馈轮次",planner_refine_count:"Planner 优化次数",
  retriever_refine_count:"Retriever 优化次数",tool_feedback_count:"工具反馈次数",
  malformed_csv:"格式异常的 CSV",empty_file:"空文件",unreadable_file:"无法读取的文件",
  malformed_jsonl:"格式异常的 JSONL",missing_file:"缺失文件"
 };
 return map[k]||k.replaceAll("_"," ").replace(/\b\w/g,c=>c.toUpperCase());
}
const statusLabel=s=>({complete:"已完成",partial:"部分数据",not_generated:"未生成"}[s]||s);
const metric=(obj,key)=>finite(obj?.[key])?obj[key]:null;
const groups={
 "效率（Efficiency）":["total_runs","token_estimator","token_saving_rate","agent_io_bytes_reduction_rate","text_agent_io_tokens","protocol_agent_io_tokens","text_per_msg_avg_tokens","protocol_per_msg_avg_tokens","latency_reduction_rate","quality_preservation_rate"],
 "通信（Communication）":["wire_bytes_reduction_rate","fair_wire_reduction_rate","text_agent_io_bytes","protocol_agent_io_bytes","protocol_total_bytes","text_wire_bytes","protocol_wire_bytes","protocol_session_dictionary_bytes","protocol_typed_envelope_bytes","protocol_typed_payload_bytes","protocol_compact_message_bytes","protocol_json_wire_bytes","protocol_state_payload_bytes"],
 "记忆（Memory）":["memory_hit_rate","memory_reused_unit_count","memory_evidence_count","memory_evidence_bytes","memory_avg_evidence_bytes_per_query","memory_avg_reused_units_per_query","memory_avg_score","memory_avg_semantic_similarity","memory_avg_tag_overlap_score"],
 "传输与运行时（Transport & Runtime）":["rust_core_enabled_runs","rust_sandbox_backend_runs","transport_send_count","transport_bytes","transport_avg_latency_ms","transport_p99_latency_ms","state_shm_transfer_count","state_shm_transfer_bytes"],
 "反馈与路由（Feedback & Routing）":["feedback_round_count","planner_refine_count","retriever_refine_count","tool_feedback_count"]
};
let suiteIndex=0,sortKey="task_id",sortAsc=true;
$("#generated").textContent=`生成时间: ${new Date(DATA.generated_at).toLocaleString()}`;
$("#completeness").textContent=`数据完整性: ${DATA.expected_sources?Math.round(DATA.completeness*100):0}% (${DATA.available_sources}/${DATA.expected_sources} 数据源)`;

function render(){
 if(!DATA.suites.length){$("#app").innerHTML=emptyPage();return}
 suiteIndex=Math.min(suiteIndex,DATA.suites.length-1);
 const suite=DATA.suites[suiteIndex],s=suite.summary||{};
 $("#app").innerHTML=`<div class="workspace">
  <aside class="rail"><div class="rail-title">测试套件（Suites）</div><div class="suite-list">${DATA.suites.map((x,i)=>suiteButton(x,i)).join("")}</div>
  <div class="legend"><span><i class="dot green"></i>已完成</span><span><i class="dot blue"></i>当前选中</span><span><i class="dot amber"></i>部分数据</span><span><i class="dot"></i>未生成</span></div></aside>
  <div class="content">
   <section class="kpis">${kpi("Token 节省",metric(s,"token_saving_rate"),"lower",true)}${kpi("Wire 减少",metric(s,"fair_wire_reduction_rate")??metric(s,"wire_bytes_reduction_rate"),"lower",true)}${kpi("延迟降低",metric(s,"latency_reduction_rate"),"lower",true)}${kpi("质量保持",metric(s,"quality_preservation_rate"),"higher",true)}${kpi("记忆命中率",metric(s,"memory_hit_rate"),"higher",true)}</section>
   <section class="section"><div class="section-head"><div><div class="section-title">模式对比汇总（Mode Comparison）</div><small>当前套件: ${esc(suite.name)}</small></div><div class="mode-key"><span><i class="key-box"></i>文本模式（Text）</span><span><i class="key-box protocol"></i>协议模式（Protocol）</span></div></div><div class="charts">${charts(suite)}</div></section>
   <section class="section"><div class="section-head"><div><div class="section-title">任务结果</div><small id="task-count"></small></div><div class="toolbar"><input id="search" class="field search" placeholder="搜索任务…" aria-label="搜索任务"><select id="mode-filter" class="field"><option value="">所有模式</option><option value="text">文本模式（Text）</option><option value="protocol">协议模式（Protocol）</option></select><button class="clear" id="clear">清除筛选</button></div></div><div class="table-wrap" id="table"></div></section>
   <section class="section"><div class="section-head"><div class="section-title">全部指标</div><small>${Object.keys(s).length} 个已记录字段</small></div><div class="metrics">${metrics(s)}</div></section>
   <section class="section"><details><summary>证据与诊断（Evidence &amp; Diagnostics）</summary><div class="evidence"><pre>${esc(suite.report||"暂无报告生成物。")}</pre><div class="diagnostics">${diagnostics(suite)}</div></div></details></section>
  </div></div>`;
 bind();
 renderTable();
}
function emptyPage(){return`<section class="section nosuites"><h2>未发现测试套件</h2><p>尚无基准测试数据。请先运行基准测试（Benchmark），然后重新生成此报告。</p><div class="kpi-value na">N/A</div><div class="kpi-note">未记录</div></section>`}
function suiteButton(x,i){return`<button class="suite ${i===suiteIndex?"active":""}" data-suite="${i}"><span class="suite-name">${esc(x.name)}</span><span class="suite-status ${x.status}"><i class="dot"></i>${statusLabel(x.status)}</span><span class="suite-meta">${x.details.length||x.configured_task_count||0} 条结果 · ${x.source_count}/3 数据源</span></button>`}
function kpi(name,value,direction,isPct){return`<div class="kpi"><div class="kpi-label">${name}</div><div class="kpi-value ${value===null?"na":""}">${value===null?"N/A":isPct?pct(value):number(value)}</div><div class="kpi-note">${value===null?"未记录":direction==="higher"?"越高越好":"越低越好"}</div></div>`}
function charts(suite){const s=suite.summary||{},defs=[
 ["Token 数",s.text_agent_io_tokens,s.protocol_agent_io_tokens,number],
 ["Agent I/O 字节",s.text_agent_io_bytes,s.protocol_agent_io_bytes,human],
 ["Wire 字节",s.text_wire_bytes,s.protocol_wire_bytes,human],
 ["延迟（Latency）",sumMode(suite.details,"text","latency_ms"),sumMode(suite.details,"protocol","latency_ms"),v=>finite(v)?`${number(v)} ms`:"N/A"]
 ];return defs.map(([name,a,b,fmt])=>chart(name,a,b,fmt)).join("")}
function sumMode(rows,mode,key){const values=rows.filter(r=>r.mode===mode).map(r=>r.metrics?.[key]).filter(finite);return values.length?values.reduce((a,b)=>a+b,0):null}
function chart(name,a,b,fmt){if(!finite(a)&&!finite(b))return`<div class="chart"><h3>${name}</h3><div class="chart-empty">暂无基准测试数据</div></div>`;const max=Math.max(a||0,b||0,1);return`<div class="chart"><h3>${name}</h3>${bar("文本模式（Text）",a,max,"",fmt)}${bar("协议模式（Protocol）",b,max,"protocol",fmt)}</div>`}
function bar(name,v,max,cls,fmt){return`<div class="bar-row"><span>${name}</span><div><div class="track"><div class="bar ${cls}" style="width:${finite(v)?Math.max(2,v/max*100):0}%"></div></div><div class="bar-value">${fmt(v)}</div></div></div>`}
function metrics(s){const seen=new Set(),content=Object.entries(groups).map(([name,keys])=>{const present=keys.filter(k=>k in s);present.forEach(k=>seen.add(k));return metricGroup(name,present,s)});const extra=Object.keys(s).filter(k=>!seen.has(k));if(extra.length)content.push(metricGroup("其他（Additional）",extra,s));return content.join("")||`<div class="empty"><strong>暂无汇总指标</strong>运行测试套件后将显示指标。</div>`}
function metricGroup(name,keys,s){return`<div class="metric-group"><h3>${esc(name)} (${keys.length})</h3>${keys.length?keys.map(k=>`<div class="metric-row"><span>${esc(zhLabel(k))}</span><code class="${s[k]===null?"na":""}">${formatMetric(k,s[k])}</code></div>`).join(""):`<div class="metric-row"><span>指标</span><code class="na">N/A</code></div>`}</div>`}
function formatMetric(k,v){if(v===null||v===undefined||v==="")return"N/A";if(typeof v!=="number")return esc(v);if(k.includes("rate"))return pct(v);if(k.includes("bytes"))return human(v);if(k.includes("latency"))return`${number(v)} ms`;return number(v)}
function diagnostics(s){if(!s.diagnostics.length)return`<div class="diagnostic" style="border-color:var(--green)"><strong>数据源已验证</strong>无解析诊断信息。</div>`;return s.diagnostics.map(d=>`<div class="diagnostic"><strong>${esc(zhLabel(d.kind))}</strong>${esc(d.source)} · ${esc(d.message)}</div>`).join("")}
function bind(){
 document.querySelectorAll("[data-suite]").forEach(b=>b.onclick=()=>{suiteIndex=Number(b.dataset.suite);render()});
 $("#search").oninput=renderTable;$("#mode-filter").onchange=renderTable;
 $("#clear").onclick=()=>{$("#search").value="";$("#mode-filter").value="";renderTable()};
}
function renderTable(){
 const suite=DATA.suites[suiteIndex],query=$("#search").value.toLowerCase(),mode=$("#mode-filter").value;
 let rows=suite.details.filter(r=>(!mode||r.mode===mode)&&JSON.stringify(r).toLowerCase().includes(query));
 rows.sort((a,b)=>compare(valueFor(a,sortKey),valueFor(b,sortKey))*(sortAsc?1:-1));
 $("#task-count").textContent=`${rows.length} / ${suite.details.length} 条结果`;
 if(!rows.length){$("#table").innerHTML=`<div class="empty"><strong>暂无基准测试数据</strong>${suite.status==="not_generated"?"此套件尚未生成。":"没有匹配当前筛选条件的结果。"}</div>`;return}
 const cols=[["task_id","任务 ID"],["group","分组"],["mode","模式"],["agent_io_tokens","Token 数"],["agent_io_bytes","Agent I/O 字节"],["wire_bytes","Wire 字节"],["latency_ms","延迟（Latency）"],["answer_quality_score","回答质量"],["memory_query_hit_count","记忆命中数"],["trace_id","Trace ID"]];
 $("#table").innerHTML=`<table class="results"><thead><tr>${cols.map(([k,n])=>`<th data-sort="${k}">${n}${sortKey===k?(sortAsc?" ↑":" ↓"):""}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(([k])=>cell(r,k)).join("")}</tr>`).join("")}</tbody></table>`;
 document.querySelectorAll("[data-sort]").forEach(h=>h.onclick=()=>{const k=h.dataset.sort;if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true}renderTable()});
}
function valueFor(r,k){return k in r?r[k]:r.metrics?.[k]}
function compare(a,b){if(a==null)return 1;if(b==null)return-1;if(typeof a==="number"&&typeof b==="number")return a-b;return String(a).localeCompare(String(b))}
function cell(r,k){const v=valueFor(r,k);let out=v==null?`<span class="na">N/A</span>`:esc(v);if(k.includes("bytes")&&finite(v))out=human(v);if(k==="latency_ms"&&finite(v))out=`${number(v)} ms`;if(k==="answer_quality_score"&&finite(v))out=number(v);return`<td class="${k==="mode"&&v==="protocol"?"mode-protocol":""}">${out}</td>`}
render();
</script>
</body>
</html>
"""
