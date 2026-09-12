"""fusion step — combine the view probabilities into one calibrated answer.

Takes the ready views, gets an unbiased P(synthetic) from each (via K-fold so we
never train and test on the same call), and learns how much to trust each view.
Works with 1, 2, or 3 views 
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import MODELS
from src.call import Call
from src.views.base import View


class Fusion:
    def __init__(self, views: list[View]):
        self.views = views
        self.meta: LogisticRegression | None = None
        self.view_names = [v.name for v in views]

    def fit(self, calls: list[Call], y: np.ndarray, n_splits: int = 5) -> np.ndarray:
        """Returns out-of-fold fused probabilities (honest, for evaluation)."""
        n = len(calls)
        oof = np.zeros((n, len(self.views)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for tr_idx, va_idx in skf.split(np.zeros(n), y):
            tr_calls = [calls[i] for i in tr_idx]
            va_calls = [calls[i] for i in va_idx]
            for j, v in enumerate(self.views):
                v.fit(tr_calls, y[tr_idx])                 # heads refit; heavy feats cached
                oof[va_idx, j] = v.proba_batch(va_calls)

        self.meta = LogisticRegression(max_iter=1000)
        self.meta.fit(oof, y)

        # refit each view on ALL labelled data for serving
        for v in self.views:
            v.fit(calls, y)
        self._save()

        return self.meta.predict_proba(oof)[:, 1]

    def proba(self, call: Call) -> tuple[float, dict]:
        per_view = {v.name: v.proba(call) for v in self.views}
        x = np.array([[per_view[v.name] for v in self.views]])
        p = float(self.meta.predict_proba(x)[0, 1])
        return p, per_view                                  # per_view = explanation

    def _save(self):
        with open(MODELS / "fusion.pkl", "wb") as f:
            pickle.dump({"meta": self.meta, "view_names": self.view_names}, f)

    def load(self) -> "Fusion":
        d = pickle.load(open(MODELS / "fusion.pkl", "rb"))
        self.meta, self.view_names = d["meta"], d["view_names"]
        for v in self.views:
            v.load()
        return self
