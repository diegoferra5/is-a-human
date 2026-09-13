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

from src.call import Call
from src.config import CALLER_CH
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
        from src.views.semantic_view import SemanticView
        caller_s = SemanticView.caller_speech_seconds(Call.from_id(r.anon_id, r.duration_s))
        f = extract(json.loads(p.read_text())["turns"], caller_speech_s=caller_s)
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
        # Raw AUC, direction included: > 0.5 means the synthetic group scores
        # higher, < 0.5 the human group. Reporting max(auc, 1-auc) was dropped
        # because it turns a null feature into a 0.52-0.56 that reads like a
        # weak signal; under label permutation a null feature lands in
        # 0.44-0.56 (95% band), so anything inside that band is nothing.
        out.append(dict(feature=name, human=h.mean(), synthetic=s.mean(),
                        d=d, auc=auc, strength=abs(auc - 0.5)))
    res = pd.DataFrame(out).sort_values("strength", ascending=False)

    NULL_BAND = (0.44, 0.56)      # 95% band of a label-permuted feature
    print(f"{'feature':<32}{'human':>9}{'synth':>9}{'d':>8}{'AUC':>7}   verdict")
    print(f"{'':<32}{'':>9}{'':>9}{'':>8}{'':>7}   (null band {NULL_BAND[0]}-{NULL_BAND[1]})")
    for _, r in res.iterrows():
        if r.strength >= 0.10:                mark = "KEEP"
        elif NULL_BAND[0] <= r.auc <= NULL_BAND[1]: mark = "null"
        else:                                 mark = "weak"
        arrow = "syn>" if r.auc > 0.5 else "hum>"
        print(f"{r.feature:<32}{r.human:9.3f}{r.synthetic:9.3f}{r.d:+8.2f}{r.auc:7.3f}   {mark} {arrow}")

    keep = res[res.strength >= 0.10].feature.tolist()
    print(f"\n{len(keep)} features with |AUC - 0.5| >= 0.10")
    (ROOT / "analysis").mkdir(exist_ok=True)
    res.to_csv(ROOT / "analysis" / "feature_scores.csv", index=False)
    df.to_csv(ROOT / "analysis" / "features_train.csv", index=False)
    print("wrote analysis/feature_scores.csv, analysis/features_train.csv")


if __name__ == "__main__":
    main()
