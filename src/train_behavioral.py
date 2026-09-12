"""Train + evaluate the behavioral-only baseline.

Reads manifest.csv, extracts turn-timing features per call, trains on `train`,
evaluates on `val`. Reports accuracy, ROC-AUC and Brier score (calibration).
Saves the fitted model + feature order to models/behavioral.pkl.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             classification_report)

import sys

from src.config import MODELS
from src.features.build import build_table


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "vad"
    print(f"turn source = {source}")
    df = build_table(source=source)
    feat_cols = [c for c in df.columns if c not in ("anon_id", "y", "split")]

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    Xtr, ytr = tr[feat_cols].values, tr.y.values
    Xva, yva = va[feat_cols].values, va.y.values

    print(f"features: {len(feat_cols)} | train n={len(tr)} "
          f"(synth {ytr.mean():.2f}) | val n={len(va)} (synth {yva.mean():.2f})")

    from xgboost import XGBClassifier
    # Small, regularized: little data + unseen test set => guard against overfit.
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        eval_metric="logloss", n_jobs=4,
        scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
    )
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xva)[:, 1]
    pred = (p >= 0.5).astype(int)
    print("\n=== VAL ===")
    print(f"accuracy : {accuracy_score(yva, pred):.3f}")
    print(f"roc_auc  : {roc_auc_score(yva, p):.3f}")
    print(f"brier    : {brier_score_loss(yva, p):.3f}  (lower=better calibrated)")
    print(classification_report(yva, pred, target_names=["human", "synthetic"]))

    imp = sorted(zip(feat_cols, model.feature_importances_),
                 key=lambda x: -x[1])[:12]
    print("top features:")
    for name, w in imp:
        print(f"  {name:24s} {w:.3f}")

    out = MODELS / "behavioral.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "feat_cols": feat_cols}, f)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
