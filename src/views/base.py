"""The contract every detection view must implement.
rules we must follow 
  - fit(calls, y): train your view. CACHE expensive per-call work (embeddings,
    transcripts) keyed by anon_id so re-fitting is cheap. Save your model to
    models/<name>.pkl.
  - proba(call) -> float in [0,1]: probability the CALLER is synthetic. --> this is the calculation of the probabilityh of the view
  - load(): restore a saved model so serve.py can use it without retraining.

"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.call import Call


class View(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, calls: list[Call], y: np.ndarray) -> None:
        """Train on labelled calls. y[i]=1 if synthetic, 0 if human."""

    @abstractmethod
    def proba(self, call: Call) -> float:
        """Return P(synthetic) for one call, in [0,1]."""

    @abstractmethod
    def load(self) -> "View":
        """Load the saved model from disk. Return self."""

    def proba_batch(self, calls: list[Call]) -> np.ndarray:
        return np.array([self.proba(c) for c in calls])
