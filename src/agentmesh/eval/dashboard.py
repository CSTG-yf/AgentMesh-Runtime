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
    generated = _generated_suite_sources(root)
    identities = set(generated)
    for name in configured:
        if not any(generated_name == name for generated_name, _track in generated):
            identities.add((name, None))
    ordered_identities = sorted(identities, key=lambda item: (item[0], item[1] or ""))
    suites: list[dict[str, object]] = []
    available_sources = 0
    expected_sources = len(ordered_identities) * 3

    for name, track in ordered_identities:
        config = configured.get(name, {})
        suite_dir = generated.get((name, track))
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
                "track": track or "not_generated",
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


def _generated_suite_sources(root: Path) -> dict[tuple[str, str | None], Path]:
    benchmark_dir = root / "runs" / "latest" / "benchmarks"
    if not benchmark_dir.exists():
        return {}
    generated: dict[tuple[str, str | None], Path] = {}
    artifact_names = {SUMMARY_FILE, DETAIL_FILE, REPORT_FILE}
    for suite_dir in benchmark_dir.iterdir():
        if not suite_dir.is_dir() or suite_dir.name.startswith("."):
            continue
        if any((suite_dir / name).exists() for name in artifact_names):
            generated[(suite_dir.name, "legacy")] = suite_dir
        for track in ["deterministic", "llm"]:
            track_dir = suite_dir / track
            if track_dir.is_dir() and any(
                (track_dir / name).exists() for name in artifact_names
            ):
                generated[(suite_dir.name, track)] = track_dir
    return generated


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
    if normalized.startswith(("{", "[")):
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            pass
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
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_html(data: dict[str, object]) -> str:
    return _HTML.replace("__DASHBOARD_DATA__", _safe_json(data))


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentMesh Benchmark</title>
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
    <h1>AgentMesh Benchmark</h1>
    <div class="meta">
      <span id="generated"></span>
      <span><i class="dot"></i>Offline report</span>
      <span id="completeness"></span>
    </div>
  </header>
  <div id="app"></div>
  <div class="footer">Self-contained report · no external runtime dependencies</div>
</main>
<script>
const DATA=__DASHBOARD_DATA__;
const $=(q,e=document)=>e.querySelector(q);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const number=v=>finite(v)?new Intl.NumberFormat("en-US",{maximumFractionDigits:2}).format(v):"N/A";
const pct=v=>finite(v)?`${(v*100).toFixed(1)}%`:"N/A";
const human=v=>{if(!finite(v))return"N/A";for(const [n,d] of [["GB",1e9],["MB",1e6],["KB",1e3]])if(Math.abs(v)>=d)return`${(v/d).toFixed(2)} ${n}`;return number(v)};
const label=k=>k.replaceAll("_"," ").replace(/\b\w/g,c=>c.toUpperCase());
const statusLabel=s=>({complete:"Completed",partial:"Partial data",not_generated:"Not generated"}[s]||s);
const metric=(obj,key)=>finite(obj?.[key])?obj[key]:null;
const groups={
 "Quality evidence":["quality_scored_runs","quality_unscored_runs","text_quality_mean","protocol_quality_mean","text_quality_pass_rate","protocol_quality_pass_rate","quality_score_delta"],
 "Efficiency":["total_runs","token_estimator","token_saving_rate","agent_io_bytes_reduction_rate","text_agent_io_tokens","protocol_agent_io_tokens","text_per_msg_avg_tokens","protocol_per_msg_avg_tokens","latency_reduction_rate"],
 "Communication":["wire_bytes_reduction_rate","fair_wire_reduction_rate","text_agent_io_bytes","protocol_agent_io_bytes","protocol_total_bytes","text_wire_bytes","protocol_wire_bytes","protocol_session_dictionary_bytes","protocol_typed_envelope_bytes","protocol_typed_payload_bytes","protocol_compact_message_bytes","protocol_json_wire_bytes","protocol_state_payload_bytes"],
 "Memory":["memory_hit_rate","memory_reused_unit_count","memory_evidence_count","memory_evidence_bytes","memory_avg_evidence_bytes_per_query","memory_avg_reused_units_per_query","memory_avg_score","memory_avg_semantic_similarity","memory_avg_tag_overlap_score"],
 "Transport & Runtime":["rust_core_enabled_runs","rust_sandbox_backend_runs","transport_send_count","transport_bytes","transport_avg_latency_ms","transport_p99_latency_ms","state_shm_transfer_count","state_shm_transfer_bytes"],
 "Feedback & Routing":["feedback_round_count","planner_refine_count","retriever_refine_count","tool_feedback_count"]
};
let suiteIndex=0,sortKey="task_id",sortAsc=true;
$("#generated").textContent=`Generated: ${new Date(DATA.generated_at).toLocaleString()}`;
$("#completeness").textContent=`Data completeness: ${DATA.expected_sources?Math.round(DATA.completeness*100):0}% (${DATA.available_sources}/${DATA.expected_sources} sources)`;

function render(){
 if(!DATA.suites.length){$("#app").innerHTML=emptyPage();return}
 suiteIndex=Math.min(suiteIndex,DATA.suites.length-1);
 const suite=DATA.suites[suiteIndex],s=suite.summary||{};
 $("#app").innerHTML=`<div class="workspace">
  <aside class="rail"><div class="rail-title">Suites</div><div class="suite-list">${DATA.suites.map((x,i)=>suiteButton(x,i)).join("")}</div>
  <div class="legend"><span><i class="dot green"></i>Completed</span><span><i class="dot blue"></i>Selected</span><span><i class="dot amber"></i>Partial data</span><span><i class="dot"></i>Not generated</span></div></aside>
  <div class="content">
   ${qualityEvidence(s)}
   <section class="kpis">${kpi("Token saving",metric(s,"token_saving_rate"),"lower",true)}${kpi("Wire reduction",metric(s,"fair_wire_reduction_rate")??metric(s,"wire_bytes_reduction_rate"),"lower",true)}${kpi("Latency reduction",metric(s,"latency_reduction_rate"),"lower",true)}${kpi("Memory hit rate",metric(s,"memory_hit_rate"),"higher",true)}</section>
   <section class="section"><div class="section-head"><div><div class="section-title">Mode comparison summary</div><small>Selected: ${esc(suite.name)} · ${esc(suite.track||"legacy")}</small></div><div class="mode-key"><span><i class="key-box"></i>Text Mode</span><span><i class="key-box protocol"></i>Protocol Mode</span></div></div><div class="charts">${charts(suite)}</div></section>
   <section class="section"><div class="section-head"><div><div class="section-title">Task results</div><small id="task-count"></small></div><div class="toolbar"><input id="search" class="field search" placeholder="Search tasks…" aria-label="Search tasks"><select id="mode-filter" class="field"><option value="">All modes</option><option value="text">Text</option><option value="protocol">Protocol</option></select><button class="clear" id="clear">Clear filters</button></div></div><div class="table-wrap" id="table"></div></section>
   <section class="section"><div class="section-head"><div class="section-title">All metrics</div><small>${Object.keys(s).length} recorded fields</small></div><div class="metrics">${metrics(s)}</div></section>
   <section class="section"><details><summary>Evidence &amp; diagnostics</summary><div class="evidence"><pre>${esc(suite.report||"No report artifact available.")}</pre><div class="diagnostics">${diagnostics(suite)}</div></div></details></section>
  </div></div>`;
 bind();
 renderTable();
}
function emptyPage(){return`<section class="section nosuites"><h2>No suites discovered</h2><p>No benchmark data is available yet. Run a benchmark and regenerate this report.</p><div class="kpi-value na">N/A</div><div class="kpi-note">Not recorded</div></section>`}
function suiteButton(x,i){return`<button class="suite ${i===suiteIndex?"active":""}" data-suite="${i}"><span class="suite-name">${esc(x.name)} · ${esc(x.track||"legacy")}</span><span class="suite-status ${x.status}"><i class="dot"></i>${statusLabel(x.status)}</span><span class="suite-meta">${x.details.length||x.configured_task_count||0} result rows · ${x.source_count}/3 sources</span></button>`}
function kpi(name,value,direction,isPct){return`<div class="kpi"><div class="kpi-label">${name}</div><div class="kpi-value ${value===null?"na":""}">${value===null?"N/A":isPct?pct(value):number(value)}</div><div class="kpi-note">${value===null?"Not recorded":direction==="higher"?"Higher is better":"Lower is better"}</div></div>`}
function qualityEvidence(s){const tm=metric(s,"text_quality_mean"),tp=metric(s,"text_quality_pass_rate"),pm=metric(s,"protocol_quality_mean"),pp=metric(s,"protocol_quality_pass_rate");return`<section class="section"><div class="section-head"><div><div class="section-title">Quality evidence</div><small>Only task-defined, versioned rules are aggregated</small></div></div><div class="kpis">${evidenceKpi("Quality scored pairs",metric(s,"quality_scored_runs"),false)}${evidenceKpi("Unscored pairs",metric(s,"quality_unscored_runs"),false)}${evidenceKpi("Text quality mean / pass rate",scorePass(tm,tp),false)}${evidenceKpi("Protocol quality mean / pass rate",scorePass(pm,pp),false)}${evidenceKpi("Quality score delta",metric(s,"quality_score_delta"),true)}</div></section>`}
function scorePass(score,passRate){return score===null||passRate===null?null:`${number(score)} / ${pct(passRate)}`}
function evidenceKpi(name,value,isPct){const missing=value===null;const shown=missing?"N/A":typeof value==="string"?value:isPct?pct(value):number(value);return`<div class="kpi"><div class="kpi-label">${name}</div><div class="kpi-value ${missing?"na":""}">${shown}</div><div class="kpi-note">${missing?"Not recorded":"Task-defined evidence"}</div></div>`}
function charts(suite){const s=suite.summary||{},defs=[
 ["Tokens",s.text_agent_io_tokens,s.protocol_agent_io_tokens,number],
 ["Agent I/O bytes",s.text_agent_io_bytes,s.protocol_agent_io_bytes,human],
 ["Wire bytes",s.text_wire_bytes,s.protocol_wire_bytes,human],
 ["Latency",sumMode(suite.details,"text","latency_ms"),sumMode(suite.details,"protocol","latency_ms"),v=>finite(v)?`${number(v)} ms`:"N/A"]
 ];return defs.map(([name,a,b,fmt])=>chart(name,a,b,fmt)).join("")}
function sumMode(rows,mode,key){const values=rows.filter(r=>r.mode===mode).map(r=>r.metrics?.[key]).filter(finite);return values.length?values.reduce((a,b)=>a+b,0):null}
function chart(name,a,b,fmt){if(!finite(a)&&!finite(b))return`<div class="chart"><h3>${name}</h3><div class="chart-empty">No benchmark data</div></div>`;const max=Math.max(a||0,b||0,1);return`<div class="chart"><h3>${name}</h3>${bar("Text Mode",a,max,"",fmt)}${bar("Protocol",b,max,"protocol",fmt)}</div>`}
function bar(name,v,max,cls,fmt){return`<div class="bar-row"><span>${name}</span><div><div class="track"><div class="bar ${cls}" style="width:${finite(v)?Math.max(2,v/max*100):0}%"></div></div><div class="bar-value">${fmt(v)}</div></div></div>`}
function metrics(s){const seen=new Set(),content=Object.entries(groups).map(([name,keys])=>{const present=keys.filter(k=>k in s);present.forEach(k=>seen.add(k));return metricGroup(name,present,s)});const extra=Object.keys(s).filter(k=>!seen.has(k));if(extra.length)content.push(metricGroup("Additional",extra,s));return content.join("")||`<div class="empty"><strong>No summary metrics</strong>Metrics will appear after this suite runs.</div>`}
function metricGroup(name,keys,s){return`<div class="metric-group"><h3>${esc(name)} (${keys.length})</h3>${keys.length?keys.map(k=>`<div class="metric-row"><span>${esc(label(k))}</span><code class="${s[k]===null?"na":""}">${formatMetric(k,s[k])}</code></div>`).join(""):`<div class="metric-row"><span>Metrics</span><code class="na">N/A</code></div>`}</div>`}
function formatMetric(k,v){if(v===null||v===undefined||v==="")return"N/A";if(typeof v!=="number")return esc(v);if(k.includes("rate"))return pct(v);if(k.includes("bytes"))return human(v);if(k.includes("latency"))return`${number(v)} ms`;return number(v)}
function diagnostics(s){if(!s.diagnostics.length)return`<div class="diagnostic" style="border-color:var(--green)"><strong>Sources verified</strong>No parsing diagnostics.</div>`;return s.diagnostics.map(d=>`<div class="diagnostic"><strong>${esc(label(d.kind))}</strong>${esc(d.source)} · ${esc(d.message)}</div>`).join("")}
function bind(){
 document.querySelectorAll("[data-suite]").forEach(b=>b.onclick=()=>{suiteIndex=Number(b.dataset.suite);render()});
 $("#search").oninput=renderTable;$("#mode-filter").onchange=renderTable;
 $("#clear").onclick=()=>{$("#search").value="";$("#mode-filter").value="";renderTable()};
}
function renderTable(){
 const suite=DATA.suites[suiteIndex],query=$("#search").value.toLowerCase(),mode=$("#mode-filter").value;
 let rows=suite.details.filter(r=>(!mode||r.mode===mode)&&JSON.stringify(r).toLowerCase().includes(query));
 rows.sort((a,b)=>compare(valueFor(a,sortKey),valueFor(b,sortKey))*(sortAsc?1:-1));
 $("#task-count").textContent=`${rows.length} of ${suite.details.length} result rows`;
 if(!rows.length){$("#table").innerHTML=`<div class="empty"><strong>No benchmark data</strong>${suite.status==="not_generated"?"This suite has not been generated.":"No rows match the current filters."}</div>`;return}
 const cols=[["task_id","Task ID"],["group","Group"],["mode","Mode"],["quality_rule_id","Quality rule"],["quality_score","Quality score"],["quality_passed","Passed"],["agent_io_tokens","Tokens"],["agent_io_bytes","Agent I/O bytes"],["wire_bytes","Wire bytes"],["latency_ms","Latency"],["memory_query_hit_count","Memory hits"],["trace_id","Trace ID"]];
 $("#table").innerHTML=`<table class="results"><thead><tr>${cols.map(([k,n])=>`<th data-sort="${k}">${n}${sortKey===k?(sortAsc?" ↑":" ↓"):""}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(([k])=>cell(r,k)).join("")}</tr>`).join("")}</tbody></table>`;
 document.querySelectorAll("[data-sort]").forEach(h=>h.onclick=()=>{const k=h.dataset.sort;if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true}renderTable()});
}
function valueFor(r,k){if(k==="quality_rule_id")return r.quality?.rule_id;if(k==="quality_score")return r.quality?.score;if(k==="quality_passed")return r.quality?.scored?r.quality?.passed:null;return k in r?r[k]:r.metrics?.[k]}
function compare(a,b){if(a==null)return 1;if(b==null)return-1;if(typeof a==="number"&&typeof b==="number")return a-b;return String(a).localeCompare(String(b))}
function cell(r,k){const v=valueFor(r,k);let out=v==null?`<span class="na">N/A</span>`:esc(v);if(k.includes("bytes")&&finite(v))out=human(v);if(k==="latency_ms"&&finite(v))out=`${number(v)} ms`;if(k==="quality_score"&&finite(v))out=number(v);return`<td class="${k==="mode"&&v==="protocol"?"mode-protocol":""}">${out}</td>`}
render();
</script>
</body>
</html>
"""
