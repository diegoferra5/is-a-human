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
# from src.views.semantic_view import SemanticView     

VIEWS = [
    BehavioralView(),
    # AcousticView(),
    # SemanticView(),
]


def report(name, y, p):
    print(f"[{name:11s}] acc={accuracy_score(y, p>=.5):.3f}  "
          f"auc={roc_auc_score(y, p):.3f}  brier={brier_score_loss(y, p):.3f}")


def _calls(df):
    calls = [Call.from_id(r.anon_id, r.duration_s) for _, r in df.iterrows()]
    y = (df.label == "synthetic").astype(int).values
    return calls, y


def main():
    man = pd.read_csv(MANIFEST)
    # Hold VAL out entirely. Fit + cross-validate on TRAIN only, then evaluate on
    # the speaker-disjoint VAL -> the honest preview of the hidden set. (Earlier
    # this folded over all 353, leaking val into training. See EXPLAINED.md.)
    tr_calls, ytr = _calls(man[man.split == "train"])
    va_calls, yva = _calls(man[man.split == "val"])

    fusion = Fusion(VIEWS)
    print(f"training {len(VIEWS)} view(s) on {len(tr_calls)} TRAIN calls "
          f"(5-fold CV), holding out {len(va_calls)} VAL...")
    oof = fusion.fit(tr_calls, ytr)     # CV within train; views refit on train

    print("\n=== metrics ===")
    report("train OOF", ytr, oof)                                   # CV estimate
    pva = np.array([fusion.proba(c)[0] for c in va_calls])
    report("VAL held-out", yva, pva)                                # honest number
    print("\nsaved models -> models/  (per-view + fusion.pkl), trained on TRAIN.")
    print("NOTE: random CV is still by-call, not by-speaker (no speaker ids) -> "
          "train OOF is optimistic; trust the VAL row.")


if __name__ == "__main__":
    main()
