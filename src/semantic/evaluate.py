"""Measure every semantic feature on the train split against the real labels.

Prints one row per feature: mean for each class, the gap, Cohen's d and the
single-feature AUC. Nothing is fitted and val is never touched.

    python src/semantic/evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.semantic.features import extract, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]


def load(split: str = "train") -> pd.DataFrame:
    man = pd.read_csv(ROOT / "dataset" / "manifest.csv")
    man = man[man.split == split]
    rows = []
    for _, r in man.iterrows():
        p = ROOT / "transcripts" / f"{r.anon_id}.json"
        if not p.exists():
            continue
        f = extract(json.loads(p.read_text())["turns"])
        f.update(call=r.anon_id, y=int(r.label == "synthetic"))
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    df = load("train")
    n_syn, n_hum = int(df.y.sum()), int((1 - df.y).sum())
    print(f"train: {n_hum} human, {n_syn} synthetic\n")

    out = []
    for name in FEATURE_NAMES:
        h = df.loc[df.y == 0, name].astype(float)
        s = df.loc[df.y == 1, name].astype(float)
        pooled = np.sqrt((h.var(ddof=1) + s.var(ddof=1)) / 2) or 1e-9
        d = (s.mean() - h.mean()) / pooled
        auc = roc_auc_score(df.y, df[name])
        out.append(dict(feature=name, human=h.mean(), synthetic=s.mean(),
                        d=d, auc=max(auc, 1 - auc), raw_auc=auc))
    res = pd.DataFrame(out).sort_values("auc", ascending=False)

    print(f"{'feature':<32}{'human':>9}{'synth':>9}{'d':>8}{'AUC':>7}   verdict")
    for _, r in res.iterrows():
        mark = "KEEP" if r.auc >= 0.60 else ("weak" if r.auc >= 0.56 else "drop")
        arrow = "syn>" if r.raw_auc > 0.5 else "hum>"
        print(f"{r.feature:<32}{r.human:9.3f}{r.synthetic:9.3f}{r.d:+8.2f}{r.auc:7.3f}   {mark} {arrow}")

    keep = res[res.auc >= 0.60].feature.tolist()
    print(f"\n{len(keep)} features at AUC >= 0.60")
    (ROOT / "analysis").mkdir(exist_ok=True)
    res.to_csv(ROOT / "analysis" / "feature_scores.csv", index=False)
    df.to_csv(ROOT / "analysis" / "features_train.csv", index=False)
    print("wrote analysis/feature_scores.csv, analysis/features_train.csv")


if __name__ == "__main__":
    main()
