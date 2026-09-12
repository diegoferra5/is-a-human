"""Self-contained HTML page for layer benchmark results."""

from __future__ import annotations

import json


def render_benchmark_page(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return _TEMPLATE.replace("__DATA__", blob)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>is-a-human benchmarks</title>
<style>
:root {
  --bg:#111318; --panel:#1a1d24; --line:#2c313c; --text:#eceef2; --muted:#8b919d;
  --ok:#2f9e44; --skip:#f59f00; --chip:#252a33;
}
* { box-sizing:border-box }
body {
  margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
}
header {
  padding:28px 28px 18px; border-bottom:1px solid var(--line);
}
.eyebrow { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 6px }
h1 { font-size:22px; margin:0; font-weight:650 }
.meta { color:var(--muted); font-size:13px; margin-top:6px }
main { padding:20px 28px 64px; max-width:1100px }
.cards {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
  gap:12px; margin-bottom:28px;
}
.card {
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px;
}
.card .k { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted) }
.card .v { font-size:26px; font-weight:650; margin:8px 0 4px; font-variant-numeric:tabular-nums }
.card .d { color:var(--muted); font-size:12px }
.ok { color:var(--ok) } .skipped { color:var(--skip) }
section { margin:0 0 28px }
h2 {
  margin:0 0 8px; font-size:12px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); display:flex; gap:8px; align-items:center;
}
h2::after { content:""; flex:1; height:1px; background:var(--line) }
.purpose { color:var(--muted); margin:0 0 12px; max-width:62ch }
.panel {
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; overflow-x:auto;
}
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums }
th, td { text-align:left; padding:8px 10px 8px 0; border-bottom:1px solid var(--line); font-size:13px }
th { color:var(--muted); font-weight:500 }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px }
.chip {
  background:var(--chip); border:1px solid var(--line); border-radius:999px;
  padding:4px 10px; font-size:12px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}
.empty { color:var(--muted) }
</style>
</head>
<body>
<header>
  <p class="eyebrow">is-a-human</p>
  <h1>Layer benchmarks</h1>
  <p class="meta" id="meta"></p>
</header>
<main>
  <div class="cards" id="cards"></div>
  <div id="sections"></div>
</main>
<script>
const DATA = __DATA__;
function fmt(n) {
  if (typeof n !== "number") return n;
  return Number.isInteger(n) ? n : n.toFixed(3);
}
document.getElementById("meta").textContent =
  DATA.generated_at + " · scored on " + DATA.split +
  (DATA.limit ? " · limit " + DATA.limit : " · full split") +
  " · " + DATA.duration_s + "s";
document.getElementById("cards").innerHTML = DATA.suites.map(s =>
  '<div class="card"><div class="k">' + s.title + "</div>" +
  '<div class="v ' + s.status + '">' + s.headline + "</div>" +
  '<div class="d">' + (s.headline_detail || "") + "</div></div>"
).join("");
document.getElementById("sections").innerHTML = DATA.suites.map(s => {
  let body = '<p class="purpose">' + s.purpose + "</p><div class='panel'>";
  if (s.status === "skipped") {
    body += '<p class="empty">' + s.reason + "</p>";
  } else if (s.table) {
    body += "<table><thead><tr>" + s.table.columns.map(c => "<th>" + c + "</th>").join("") +
      "</tr></thead><tbody>" + s.table.rows.map(r =>
        "<tr>" + r.map(c => "<td>" + fmt(c) + "</td>").join("") + "</tr>"
      ).join("") + "</tbody></table>";
    if (s.top_separators && s.top_separators.length) {
      body += '<div class="chips">' + s.top_separators.map(item =>
        '<span class="chip">' + item.feature + " d=" + item.effect_size + "</span>"
      ).join("") + "</div>";
    }
  }
  body += "</div>";
  return "<section><h2>" + s.title + " · " + s.status + "</h2>" + body + "</section>";
}).join("");
</script>
</body>
</html>
"""
