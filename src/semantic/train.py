"""Train the semantic view on the train split and save it.

    python -m src.semantic.train            # all 12 features
    python -m src.semantic.train --safe     # compact 5-feature variant -> semantic_safe.pkl

Prints the learned weights and a worked example so the model is inspectable.
Val is not touched here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.call import Call
from src.views.semantic_view import SemanticView

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", action="store_true",
                    help="drop every feature that counts words")
    args = ap.parse_args()

    man = pd.read_csv(ROOT / "dataset" / "manifest.csv")
    man = man[(man.split == "train")
              & man.anon_id.map(lambda i: (ROOT / "transcripts" / f"{i}.json").exists())]
    calls = [Call.from_id(r.anon_id, r.duration_s) for _, r in man.iterrows()]
    y = (man.label == "synthetic").to_numpy(int)

    view = SemanticView(safe_only=args.safe)
    X = np.array([view._features(c) for c in calls])
    print(f"{len(calls)} train calls, {len(view.features)} features "
          f"({'safe only' if args.safe else 'all'})\n")

    # Score the *same* pipeline the view will fit, so the printed numbers
    # describe the model that actually gets saved.
    pipe = SemanticView.make_pipeline()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
    print(f"5-fold on train:  AUC {auc:.3f}   acc {100*acc:.1f}%")
    print(f"baseline (always synthetic): {100*max(y.mean(), 1-y.mean()):.1f}%\n")

    view.fit(calls, y)

    clf = view.model[-1]
    print("learned weights (+ pushes toward synthetic):")
    for name, w in sorted(zip(view.features, clf.coef_[0]),
                          key=lambda t: abs(t[1]), reverse=True):
        print(f"  {w:+6.2f}  {name}")

    # one worked example, so the explain() path is exercised and visible
    probs = np.array([view.proba(c) for c in calls])
    i = int(np.argmax(probs))
    print(f"\nmost-synthetic-looking train call ({calls[i].anon_id}, "
          f"true label {'synthetic' if y[i] else 'human'}, p={probs[i]:.3f}):")
    for name, val, contrib in view.explain(calls[i])[:5]:
        print(f"  {contrib:+6.2f}  {name} = {val:g}")

    print(f"\nsaved {view.pkl}")


if __name__ == "__main__":
    main()
