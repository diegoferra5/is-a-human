"""Pytest plugin: write a self-contained Playwright-style HTML report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SUITES = ("vad", "acoustic", "semantic", "behavioral")
DEFAULT_REPORT = Path("reports/test-results/index.html")


def pytest_addoption(parser):
    group = parser.getgroup("html-report")
    group.addoption(
        "--html-report",
        action="store",
        default=str(DEFAULT_REPORT),
        help="Path for the HTML test report (default: reports/test-results/index.html)",
    )
    group.addoption(
        "--no-html-report",
        action="store_true",
        default=False,
        help="Disable the HTML test report",
    )


def pytest_configure(config):
    if config.getoption("--no-html-report"):
        return
    reporter = HtmlReporter(Path(config.getoption("--html-report")))
    config.pluginmanager.register(reporter, name="is_a_human_html_reporter")
    config._html_reporter = reporter  # type: ignore[attr-defined]


def render_report(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return _TEMPLATE.replace("__DATA__", blob)


def _suite_for(item) -> str:
    names = {mark.name for mark in item.iter_markers()}
    for suite in SUITES:
        if suite in names:
            return suite
    return "other"


def _metrics(report) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for key, value in getattr(report, "user_properties", []) or []:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[str(key)] = round(float(value), 4)
        else:
            values[str(key)] = str(value)
    return values


class HtmlReporter:
    def __init__(self, output: Path):
        self.output = output
        self._started = datetime.now(timezone.utc)
        self._by_node: dict[str, dict] = {}
        self._suites: dict[str, str] = {}

    def pytest_itemcollected(self, item) -> None:
        self._suites[item.nodeid] = _suite_for(item)

    def pytest_runtest_logreport(self, report) -> None:
        current = self._by_node.get(report.nodeid)
        if current is None:
            current = {
                "nodeid": report.nodeid,
                "name": report.nodeid.split("::", 1)[-1],
                "file": report.nodeid.split("::", 1)[0],
                "suite": self._suites.get(report.nodeid, "other"),
                "outcome": "passed",
                "duration_s": 0.0,
                "error": "",
                "metrics": {},
            }
            self._by_node[report.nodeid] = current

        current["duration_s"] = round(current["duration_s"] + float(report.duration), 4)
        metrics = _metrics(report)
        if metrics:
            current["metrics"].update(metrics)

        failed = report.failed
        skipped = report.skipped
        if report.when == "call" or (report.when == "setup" and (failed or skipped)):
            if skipped:
                current["outcome"] = "skipped"
                current["error"] = str(report.longrepr) if report.longrepr else ""
            elif failed:
                current["outcome"] = "failed"
                current["error"] = str(report.longrepr) if report.longrepr else ""
            elif report.when == "call":
                current["outcome"] = "passed"

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        tests = list(self._by_node.values())
        counts = {"passed": 0, "failed": 0, "skipped": 0}
        for test in tests:
            counts[test["outcome"]] = counts.get(test["outcome"], 0) + 1
        elapsed = (datetime.now(timezone.utc) - self._started).total_seconds()
        payload = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "duration_s": round(elapsed, 2),
            "exitstatus": int(exitstatus),
            "counts": counts,
            "tests": tests,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(render_report(payload), encoding="utf-8")

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config) -> None:
        path = self.output.resolve()
        terminalreporter.write_sep("=", "HTML report")
        terminalreporter.write_line(f"Open: {path.as_uri()}")
        terminalreporter.write_line(f"File: {path}")


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>is-a-human test report</title>
<style>
:root {
  --bg:#111318; --panel:#1a1d24; --line:#2c313c; --text:#eceef2; --muted:#8b919d;
  --pass:#2f9e44; --fail:#e03131; --skip:#f59f00; --chip:#252a33;
}
* { box-sizing:border-box }
body {
  margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
}
header {
  padding:28px 28px 18px; border-bottom:1px solid var(--line);
  display:flex; flex-wrap:wrap; gap:18px; align-items:flex-end; justify-content:space-between;
}
.eyebrow { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 6px }
h1 { font-size:22px; margin:0; font-weight:650 }
.meta { color:var(--muted); font-size:13px; margin-top:6px }
.counts { display:flex; gap:8px; flex-wrap:wrap }
.stat {
  background:var(--chip); border:1px solid var(--line); border-radius:999px;
  padding:6px 12px; font-variant-numeric:tabular-nums;
}
.stat b { font-weight:650 }
.pass { color:var(--pass) } .fail { color:var(--fail) } .skip { color:var(--skip) }
main { padding:20px 28px 64px; max-width:1100px }
.filters { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px }
button {
  background:var(--chip); color:var(--text); border:1px solid var(--line);
  border-radius:8px; padding:6px 10px; cursor:pointer; font:inherit;
}
button[aria-pressed="true"] { border-color:#6b8aff; background:#20263a }
.group { margin:0 0 22px }
.group h2 {
  margin:0 0 8px; font-size:12px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); display:flex; gap:8px; align-items:center;
}
.group h2::after { content:""; flex:1; height:1px; background:var(--line) }
.row {
  background:var(--panel); border:1px solid var(--line); border-radius:10px;
  margin:0 0 8px; overflow:hidden;
}
.row.failed { border-color:#5c2626 }
summary {
  list-style:none; cursor:pointer; display:flex; gap:10px; align-items:center;
  padding:10px 12px;
}
summary::-webkit-details-marker { display:none }
.dot { width:8px; height:8px; border-radius:50%; flex:0 0 auto }
.dot.passed { background:var(--pass) }
.dot.failed { background:var(--fail) }
.dot.skipped { background:var(--skip) }
.title { flex:1; min-width:0 }
.title .name { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px }
.title .file { color:var(--muted); font-size:12px; margin-top:2px }
.dur { color:var(--muted); font-variant-numeric:tabular-nums; font-size:12px }
.detail { border-top:1px solid var(--line); padding:10px 12px 12px; }
pre {
  margin:0; white-space:pre-wrap; word-break:break-word;
  font:12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color:#ffc9c9;
}
table { border-collapse:collapse; width:auto; margin-top:8px; font-size:12px }
th, td { text-align:left; padding:4px 10px 4px 0; border-bottom:1px solid var(--line) }
th { color:var(--muted); font-weight:500 }
.empty { color:var(--muted); padding:24px 0 }
</style>
</head>
<body>
<header>
  <div>
    <p class="eyebrow">is-a-human</p>
    <h1>Test report</h1>
    <p class="meta" id="meta"></p>
  </div>
  <div class="counts" id="counts"></div>
</header>
<main>
  <div class="filters" id="filters"></div>
  <div id="groups"></div>
</main>
<script>
const DATA = __DATA__;
const SUITES = ["vad","acoustic","semantic","behavioral","other"];
const LABELS = {vad:"VAD", acoustic:"Acoustic", semantic:"Semantic", behavioral:"Behavioural", other:"Other"};
let outcomeFilter = "all";
let suiteFilter = "all";

function fmtDur(s) {
  if (s < 1) return Math.round(s * 1000) + "ms";
  return s.toFixed(2) + "s";
}
function setMeta() {
  const n = DATA.tests.length;
  document.getElementById("meta").textContent =
    DATA.generated_at + " · " + n + " tests · " + fmtDur(DATA.duration_s);
  const c = DATA.counts;
  document.getElementById("counts").innerHTML =
    '<span class="stat pass"><b>' + (c.passed||0) + "</b> passed</span>" +
    '<span class="stat fail"><b>' + (c.failed||0) + "</b> failed</span>" +
    '<span class="stat skip"><b>' + (c.skipped||0) + "</b> skipped</span>";
}
function filters() {
  const outcomes = ["all","passed","failed","skipped"];
  const suites = ["all"].concat(SUITES);
  const html = outcomes.map(v =>
    '<button data-kind="outcome" data-val="' + v + '" aria-pressed="' + (outcomeFilter===v) + '">' + v + "</button>"
  ).join("") + suites.map(v =>
    '<button data-kind="suite" data-val="' + v + '" aria-pressed="' + (suiteFilter===v) + '">' + (LABELS[v]||v) + "</button>"
  ).join("");
  const el = document.getElementById("filters");
  el.innerHTML = html;
  el.querySelectorAll("button").forEach(btn => btn.onclick = () => {
    if (btn.dataset.kind === "outcome") outcomeFilter = btn.dataset.val;
    else suiteFilter = btn.dataset.val;
    render();
  });
}
function visible(t) {
  if (outcomeFilter !== "all" && t.outcome !== outcomeFilter) return false;
  if (suiteFilter !== "all" && t.suite !== suiteFilter) return false;
  return true;
}
function metricsTable(m) {
  const keys = Object.keys(m || {});
  if (!keys.length) return "";
  return "<table><tbody>" + keys.map(k =>
    "<tr><th>" + k + "</th><td>" + m[k] + "</td></tr>"
  ).join("") + "</tbody></table>";
}
function render() {
  setMeta();
  filters();
  const tests = DATA.tests.filter(visible);
  const by = {};
  for (const t of tests) (by[t.suite] ||= []).push(t);
  const root = document.getElementById("groups");
  if (!tests.length) { root.innerHTML = '<p class="empty">No tests match these filters.</p>'; return; }
  root.innerHTML = SUITES.filter(s => by[s]).map(suite => {
    const rows = by[suite].map(t => {
      const err = t.error ? "<pre>" + t.error.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])) + "</pre>" : "";
      const extra = err + metricsTable(t.metrics);
      const open = t.outcome === "failed" ? " open" : "";
      return '<details class="row ' + t.outcome + '"' + open + "><summary>" +
        '<span class="dot ' + t.outcome + '"></span><div class="title">' +
        '<div class="name">' + t.name + "</div>" +
        '<div class="file">' + t.file + "</div></div>" +
        '<span class="dur">' + fmtDur(t.duration_s) + "</span></summary>" +
        '<div class="detail">' + (extra || "<span class='empty'>No extra output</span>") +
        "</div></details>";
    }).join("");
    return '<section class="group"><h2>' + LABELS[suite] + " · " + by[suite].length + "</h2>" + rows + "</section>";
  }).join("");
}
render();
</script>
</body>
</html>
"""
