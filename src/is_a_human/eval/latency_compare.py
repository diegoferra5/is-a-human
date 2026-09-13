"""Before/after latency and verdict comparison for the live extract path."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from is_a_human.analysis.features import extract_call_features_timed
from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, load_tandem
from is_a_human.dataset.loader import iter_split
from is_a_human.pipeline import process_call

DEFAULT_OUTPUT = Path("reports/latency/index.html")
LAYERS = ("vad", "acoustic", "semantic", "behavioral")


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = sorted(zip(scores, labels))
    ranks: dict[int, float] = {}
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2
        i = j
    pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y == 1]
    n_pos, n_neg = len(pos), len(pairs) - len(pos)
    if not n_pos or not n_neg:
        return None
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _load_transcript(anon_id: str, transcripts_dir: Path) -> list[dict] | None:
    path = transcripts_dir / f"{anon_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns = payload.get("turns")
    return turns if isinstance(turns, list) else None


def _summarize(rows: list[dict], probability_key: str, latency_key: str) -> dict:
    answered = [row for row in rows if probability_key in row]
    tp = sum(1 for row in answered if row["label"] == "synthetic" and row[probability_key] >= 0.5)
    tn = sum(1 for row in answered if row["label"] == "human" and row[probability_key] < 0.5)
    n_syn = sum(1 for row in answered if row["label"] == "synthetic")
    n_hum = sum(1 for row in answered if row["label"] == "human")
    tpr = tp / n_syn if n_syn else None
    tnr = tn / n_hum if n_hum else None
    scores = [row[probability_key] for row in answered]
    labels = [1 if row["label"] == "synthetic" else 0 for row in answered]
    latencies = [row[latency_key] for row in answered]
    prefix = "before" if latency_key.startswith("before") else "after"
    layer_means = {}
    for layer in (*LAYERS, "acoustic_extras"):
        values = [
            row[f"{prefix}_{layer}_ms"]
            for row in answered
            if row.get(f"{prefix}_{layer}_ms") is not None
        ]
        layer_means[layer] = sum(values) / len(values) if values else None
    return {
        "calls": len(answered),
        "accuracy": (tp + tn) / len(answered) if answered else None,
        "tpr_synthetic": tpr,
        "tnr_human": tnr,
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "auc": _auc(scores, labels),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "mean_layer_ms": layer_means,
    }


def _layer_total(timings: dict[str, float | None], include_extras: bool) -> float:
    total = 0.0
    for key in LAYERS:
        value = timings.get(key)
        if value is not None:
            total += value
    extras = timings.get("acoustic_extras")
    if include_extras and extras:
        total += extras
    return total


def compare_call(model, sample, transcript_turns: list[dict] | None) -> dict:
    started = perf_counter()
    result = process_call(sample.ch0_caller, sample.ch1_agent, sample.sample_rate)
    vad_ms = (perf_counter() - started) * 1000

    common = dict(
        anon_id=sample.anon_id,
        label=sample.label,
        split=sample.split,
        ch0_caller=sample.ch0_caller,
        ch1_agent=sample.ch1_agent,
        sample_rate=sample.sample_rate,
        pipeline_result=result,
        transcript_turns=transcript_turns,
    )
    before_features, before_t = extract_call_features_timed(**common, heavy=True)
    after_features, after_t = extract_call_features_timed(**common, heavy=False)
    before_t["vad"] = vad_ms
    after_t["vad"] = vad_ms

    before_p, before_views = model.predict(before_features)
    after_p, after_views = model.predict(after_features)

    return {
        "call_id": sample.anon_id,
        "label": sample.label,
        "duration_s": sample.duration_s,
        "semantic_ran": transcript_turns is not None,
        "before_p": before_p,
        "after_p": after_p,
        "before_acoustic_p": before_views["acoustic"],
        "after_acoustic_p": after_views["acoustic"],
        "before_behavioral_p": before_views["behavioral"],
        "after_behavioral_p": after_views["behavioral"],
        "p_delta": abs(after_p - before_p),
        "verdict_changed": (before_p >= 0.5) != (after_p >= 0.5),
        "before_ms": _layer_total(before_t, include_extras=True),
        "after_ms": _layer_total(after_t, include_extras=False),
        "before_vad_ms": before_t["vad"],
        "after_vad_ms": after_t["vad"],
        "before_acoustic_ms": before_t["acoustic"],
        "after_acoustic_ms": after_t["acoustic"],
        "before_semantic_ms": before_t["semantic"],
        "after_semantic_ms": after_t["semantic"],
        "before_behavioral_ms": before_t["behavioral"],
        "after_behavioral_ms": after_t["behavioral"],
        "before_acoustic_extras_ms": before_t["acoustic_extras"],
        "after_acoustic_extras_ms": after_t["acoustic_extras"],
    }


def run_latency_compare(
    *,
    split: str = "val",
    limit: int | None = None,
    transcripts_dir: Path | None = None,
) -> dict:
    model = load_tandem(DEFAULT_MODEL_PATH)
    transcript_root = transcripts_dir or Path("transcripts")
    rows = []
    for index, sample in enumerate(iter_split(split, load_audio=True), start=1):
        if limit is not None and index > limit:
            break
        turns = _load_transcript(sample.anon_id, transcript_root)
        row = compare_call(model, sample, turns)
        rows.append(row)
        semantic = "skip" if row["before_semantic_ms"] is None else f"{row['before_semantic_ms']:.1f}ms"
        print(
            f"{index:3d} {sample.anon_id} before={row['before_ms']:.0f}ms after={row['after_ms']:.0f}ms "
            f"vad={row['before_vad_ms']:.1f} acoustic={row['before_acoustic_ms']:.1f}->{row['after_acoustic_ms']:.1f} "
            f"semantic={semantic} behavioral={row['before_behavioral_ms']:.1f} "
            f"extras={row['before_acoustic_extras_ms']:.0f} p={row['before_p']:.3f}->{row['after_p']:.3f}",
            flush=True,
        )

    flipped = sum(1 for row in rows if row["verdict_changed"])
    matched = sum(1 for row in rows if row["p_delta"] < 1e-9)
    return {
        "split": split,
        "fusion": model.fusion_type,
        "path_before": "full (VAD + acoustic + extras + behavioural + semantic if present)",
        "path_after": "live (same fusion; skip unused formant/pitch/interaction extras)",
        "before": _summarize(rows, "before_p", "before_ms"),
        "after": _summarize(rows, "after_p", "after_ms"),
        "p_exact_match": matched,
        "verdicts_flipped": flipped,
        "mean_p_delta": sum(row["p_delta"] for row in rows) / len(rows) if rows else None,
        "semantic_calls": sum(1 for row in rows if row["semantic_ran"]),
        "rows": rows,
    }


def render_latency_page(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return _TEMPLATE.replace("__DATA__", blob)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>is-a-human latency</title>
<style>
:root { --bg:#111318; --panel:#1a1d24; --line:#2c313c; --text:#eceef2; --muted:#8b919d; --ok:#2f9e44; --bad:#fa5252; }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif, system-ui, sans-serif }
header { padding:28px 28px 18px; border-bottom:1px solid var(--line) }
h1 { font-size:22px; margin:0 }
.meta { color:var(--muted); margin-top:6px }
main { padding:20px 28px 64px; max-width:1200px }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px }
.k { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted) }
.v { font-size:22px; font-weight:650; margin:8px 0 4px; font-variant-numeric:tabular-nums }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums }
th, td { text-align:left; padding:8px 8px 8px 0; border-bottom:1px solid var(--line); font-size:12px }
th { color:var(--muted); font-weight:500 }
.flip { color:var(--bad) }
.ok { color:var(--ok) }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; overflow-x:auto }
</style>
</head>
<body>
<header>
  <h1>Live extract: before / after</h1>
  <p class="meta" id="meta"></p>
</header>
<main>
  <div class="cards" id="cards"></div>
  <div class="panel">
    <table>
      <thead><tr>
        <th>call</th><th>label</th><th>before p</th><th>after p</th>
        <th>before ms</th><th>after ms</th>
        <th>VAD</th><th>Acoustic</th><th>Semantic</th><th>Behavioural</th><th>Extras</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</main>
<script>
const data = __DATA__;
const fmt = (v, d=3) => v == null ? "skip" : Number(v).toFixed(d);
document.getElementById("meta").textContent =
  data.split + " · " + data.fusion + " · flipped " + data.verdicts_flipped +
  " · p exact " + data.p_exact_match + " · semantic " + data.semantic_calls;
const b = data.before, a = data.after;
const cards = [
  ["Before bal. acc", b.balanced_accuracy, 3],
  ["After bal. acc", a.balanced_accuracy, 3],
  ["Before mean ms", b.mean_latency_ms, 1],
  ["After mean ms", a.mean_latency_ms, 1],
  ["Before AUC", b.auc, 3],
  ["After AUC", a.auc, 3],
  ["Mean VAD ms", a.mean_layer_ms.vad, 1],
  ["Mean acoustic ms after", a.mean_layer_ms.acoustic, 1],
  ["Mean behavioural ms", a.mean_layer_ms.behavioral, 1],
  ["Mean extras ms before", b.mean_layer_ms.acoustic_extras, 1],
];
document.getElementById("cards").innerHTML = cards.map(([k,v,d]) =>
  `<div class="card"><div class="k">${k}</div><div class="v">${fmt(v,d)}</div></div>`
).join("");
document.getElementById("rows").innerHTML = data.rows.map(r =>
  `<tr class="${r.verdict_changed ? "flip" : ""}">
    <td>${r.call_id}</td><td>${r.label}</td>
    <td>${fmt(r.before_p)}</td><td>${fmt(r.after_p)}</td>
    <td>${fmt(r.before_ms,1)}</td><td class="ok">${fmt(r.after_ms,1)}</td>
    <td>${fmt(r.before_vad_ms,1)}</td>
    <td>${fmt(r.after_acoustic_ms,1)}</td>
    <td>${fmt(r.before_semantic_ms,1)}</td>
    <td>${fmt(r.after_behavioral_ms,1)}</td>
    <td>${fmt(r.before_acoustic_extras_ms,1)}</td>
  </tr>`
).join("");
</script>
</body>
</html>
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--transcripts-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = run_latency_compare(
        split=args.split,
        limit=args.limit,
        transcripts_dir=args.transcripts_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_latency_page(payload), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("before:", payload["before"])
    print("after: ", payload["after"])
    print("flipped verdicts:", payload["verdicts_flipped"])
    print("fused p exact match:", payload["p_exact_match"])
    print(f"Open: {args.output.resolve().as_uri()}")


if __name__ == "__main__":
    main()
