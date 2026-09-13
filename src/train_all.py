"""Train every ready view + the fusion, and report honest metrics.

Enable a view by uncommenting
Run:  python3 -m src.train_all
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from src.config import MANIFEST
from src.call import Call
from src.fusion import Fusion
from src.views.behavioral_view import BehavioralView
# from src.views.acoustic_view import AcousticView     
from src.views.semantic_view import SemanticView

VIEWS = [
    BehavioralView(),
    # AcousticView(),
    SemanticView(),
]


def report(name, y, p):
    print(f"[{name:11s}] acc={accuracy_score(y, p>=.5):.3f}  "
          f"auc={roc_auc_score(y, p):.3f}  brier={brier_score_loss(y, p):.3f}")


def main():
    man = pd.read_csv(MANIFEST)
    # train only: the splits are speaker-disjoint on purpose. Folding over all
    # 353 would put val speakers into training and inflate the OOF metric.
    man = man[man.split == "train"].reset_index(drop=True)
    calls = [Call.from_id(r.anon_id, r.duration_s) for _, r in man.iterrows()]
    y = (man.label == "synthetic").astype(int).values

    fusion = Fusion(VIEWS)
    print(f"training {len(VIEWS)} view(s) on {len(calls)} calls (5-fold OOF)...")
    oof = fusion.fit(calls, y)

    print("\n=== honest out-of-fold metrics ===")
    report("FUSION", y, oof)
    print("\nsaved models -> models/  (per-view + fusion.pkl)")


if __name__ == "__main__":
    main()
