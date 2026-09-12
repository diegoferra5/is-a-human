"""Validate the VAD two ways:
  1) frame-level speech/non-speech agreement vs provided turns.json (per channel)
  2) END-TO-END: rebuild behavioral features from VAD turns, run the trained
     model, and compare val metrics to the provided-turns model.
This tells us whether serve-time (VAD) features match train-time features.
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from src.config import MANIFEST, TURNS, AUDIO, MODELS
from src.vad import extract_turns
from src.features.behavioral import extract

FRAME_S = 0.02


def _mask(turns, ch, dur):
    n = int(dur / FRAME_S) + 1
    m = np.zeros(n, bool)
    for t in turns:
        if t["channel"] == ch:
            a, b = int(t["start"] / FRAME_S), int(t["end"] / FRAME_S)
            m[a:min(b, n)] = True
    return m


def frame_agreement(sample):
    ious, accs = [], []
    for aid, dur in sample:
        gt = json.load(open(TURNS / f"{aid}.json"))["turns"]
        pred = extract_turns(AUDIO / f"{aid}.wav")["turns"]
        for ch in (0, 1):
            g, p = _mask(gt, ch, dur), _mask(pred, ch, dur)
            n = min(len(g), len(p)); g, p = g[:n], p[:n]
            inter = (g & p).sum(); union = (g | p).sum()
            ious.append(inter / union if union else 1.0)
            accs.append((g == p).mean())
    return np.mean(accs), np.mean(ious)


def end_to_end():
    man = pd.read_csv(MANIFEST)
    rows = []
    for _, r in man.iterrows():
        feats = extract_turns_to_features(r.anon_id, r.duration_s)
        feats.update(y=1 if r.label == "synthetic" else 0, split=r.split)
        rows.append(feats)
    df = pd.DataFrame(rows)

    d = pickle.load(open(MODELS / "behavioral.pkl", "rb"))
    model, cols = d["model"], d["feat_cols"]
    va = df[df.split == "val"]
    X, y = va[cols].values, va.y.values
    p = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, (p >= .5)),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def extract_turns_to_features(aid, dur):
    turns = extract_turns(AUDIO / f"{aid}.wav")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(turns, f); tmp = f.name
    feats = extract(Path(tmp), duration_s=dur)
    Path(tmp).unlink()
    return feats


if __name__ == "__main__":
    man = pd.read_csv(MANIFEST)
    sample = list(man.sample(30, random_state=0)[["anon_id", "duration_s"]]
                  .itertuples(index=False, name=None))
    acc, iou = frame_agreement(sample)
    print(f"[frame-level vs provided turns]  accuracy={acc:.3f}  IoU={iou:.3f}")
    print("[end-to-end: VAD turns -> trained model, VAL]")
    for k, v in end_to_end().items():
        print(f"   {k:9s} {v:.3f}")
