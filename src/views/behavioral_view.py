"""behavioral view

Signal: WHO talks WHEN. Response latency, turn lengths, overlaps, talk ratio.
No audio content, no words -> speaker-independent -> generalizes to unseen voices.

This is the reference implementation. the other 2 views copy this implementation
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from src.config import MODELS
from src.features.behavioral import extract
from src.views.base import View
from src.call import Call


class BehavioralView(View):
    name = "behavioral"

    def __init__(self):
        self.model = None
        self.feat_cols: list[str] = []

    # -- turn a call into the 51 timing features --
    def _features(self, call: Call) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(call.turns(), f)
            tmp = f.name
        feats = extract(Path(tmp), duration_s=call.duration_s)
        Path(tmp).unlink()
        return feats

    def fit(self, calls: list[Call], y: np.ndarray) -> None:
        rows = [self._features(c) for c in calls]
        self.feat_cols = sorted(rows[0].keys())
        X = np.array([[r[c] for c in self.feat_cols] for r in rows])

        from xgboost import XGBClassifier
        pos = max((y == 1).sum(), 1)
        self.model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            eval_metric="logloss", n_jobs=4,
            scale_pos_weight=(y == 0).sum() / pos,
        )
        self.model.fit(X, y)
        self._save()

    def proba(self, call: Call) -> float:
        r = self._features(call)
        X = np.array([[r[c] for c in self.feat_cols]])
        return float(self.model.predict_proba(X)[0, 1])

    def _save(self):
        with open(MODELS / f"{self.name}.pkl", "wb") as f:
            pickle.dump({"model": self.model, "feat_cols": self.feat_cols}, f)

    def load(self) -> "BehavioralView":
        d = pickle.load(open(MODELS / f"{self.name}.pkl", "rb"))
        self.model, self.feat_cols = d["model"], d["feat_cols"]
        return self
