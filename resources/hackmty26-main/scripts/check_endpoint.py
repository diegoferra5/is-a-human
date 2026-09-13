"""Send dataset calls to a /detect endpoint exactly the way the judge does, and score the answers.

    python scripts/check_endpoint.py --url http://localhost:8000/detect
    python scripts/check_endpoint.py --url https://team.example.com/detect --n 20 --split val

Request body (JSON):  {"call_id": "...", "audio_base64": "<base64 of the WAV file bytes>", "sample_rate": 8000, "channels": 2}
Expected response:    {"is_synthetic": true|false, "confidence": 0.0-1.0}   (confidence optional)

Standard library only. Uses manifest.csv + audio/ from the repo root by default.
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_manifest(path, split):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows


def post_call(url, audio_dir, row, timeout):
    path = os.path.join(audio_dir, row["anon_id"] + ".wav")
    body = json.dumps(
        {
            "call_id": row["anon_id"],
            "audio_base64": base64.b64encode(open(path, "rb").read()).decode("ascii"),
            "sample_rate": 8000,
            "channels": 2,
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "latency_s": time.perf_counter() - started}
    except Exception as e:  # timeout, connection refused, DNS
        return {"error": type(e).__name__, "latency_s": time.perf_counter() - started}
    latency = time.perf_counter() - started
    try:
        data = json.loads(raw)
    except ValueError:
        return {"error": f"HTTP {status}, body is not JSON", "latency_s": latency}
    if not isinstance(data, dict) or not isinstance(data.get("is_synthetic"), bool):
        return {"error": "missing boolean is_synthetic", "latency_s": latency}
    conf = data.get("confidence")
    if conf is not None and not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
        return {"error": "confidence must be a number between 0 and 1", "latency_s": latency}
    return {"is_synthetic": data["is_synthetic"], "confidence": conf, "latency_s": latency}


def auc(scores, labels):
    """ROC AUC by rank; scores = probability of synthetic, labels = 1 for synthetic."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
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


def score(results):
    answered = [r for r in results if "error" not in r]
    tp = sum(1 for r in answered if r["label"] == "synthetic" and r["is_synthetic"])
    tn = sum(1 for r in answered if r["label"] == "human" and not r["is_synthetic"])
    n_syn = sum(1 for r in answered if r["label"] == "synthetic")
    n_hum = sum(1 for r in answered if r["label"] == "human")
    out = {
        "calls": len(results),
        "answered": len(answered),
        "errors": len(results) - len(answered),
        "accuracy": (tp + tn) / len(answered) if answered else None,
        "tpr_synthetic": tp / n_syn if n_syn else None,
        "tnr_human": tn / n_hum if n_hum else None,
        "mean_latency_s": sum(r["latency_s"] for r in results) / len(results) if results else None,
        "max_latency_s": max(r["latency_s"] for r in results) if results else None,
    }
    out["balanced_accuracy"] = (out["tpr_synthetic"] + out["tnr_human"]) / 2 if n_syn and n_hum else None
    with_conf = [r for r in answered if r["confidence"] is not None]
    if len(with_conf) == len(answered) and answered:
        probs = [r["confidence"] if r["is_synthetic"] else 1 - r["confidence"] for r in with_conf]
        ys = [1 if r["label"] == "synthetic" else 0 for r in with_conf]
        out["auc"] = auc(probs, ys)
        out["brier"] = sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(ys)
    else:
        out["auc"] = out["brier"] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="your /detect endpoint")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    ap.add_argument("--audio-dir", default=os.path.join(ROOT, "audio"))
    ap.add_argument("--split", default="val", choices=["train", "val", "all", "hidden"])
    ap.add_argument("--n", type=int, default=10, help="how many calls to send (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds per call, same as the judge")
    ap.add_argument("--out", help="write per-call results and the summary to this JSON file")
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.split)
    if not rows:
        sys.exit(f"no rows for split {args.split!r} in {args.manifest}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.n:
        rows = rows[: args.n]
    results = []
    for i, row in enumerate(rows, 1):
        r = post_call(args.url, args.audio_dir, row, args.timeout)
        r.update({"call_id": row["anon_id"], "label": row["label"]})
        verdict = r.get("error") or ("synthetic" if r["is_synthetic"] else "human")
        mark = "" if "error" in r else ("ok " if verdict == row["label"] else "MISS")
        conf = "" if r.get("confidence") is None else f" conf={r['confidence']:.2f}"
        print(f"{i:3d}/{len(rows)} {row['anon_id']} truth={row['label']:9s} got={verdict:20s} {mark}{conf} {r['latency_s']:.2f}s", flush=True)
        results.append(r)
    summary = score(results)
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    if args.out:
        json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
