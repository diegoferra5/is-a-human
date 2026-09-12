"""semantic view

Signal: what the caller says and whether it makes sense. The bank agent
deliberately asks nonsense questions, repeats, interrupts. A scripted LLM bot
handles these differently than a confused human.

Transcripts are produced ahead of time by src/transcribe.py (mlx-whisper,
large-v3-turbo, es) and read from transcripts/<anon_id>.json:

    {"call_id": ..., "turns": [{"channel": 0, "start": 10.5, "end": 20.2, "text": "..."}]}

Turn order and timestamps are kept on purpose: the probes we look for are
*pairs* -- something the agent said, and how the caller answered right after.
Flattening each side into one blob of text would destroy that.

Plan (transcript -> probe features -> classifier):
  1. Read the transcript (both channels, in order).
  2. Locate probe moments in the AGENT channel -- the script is near-identical
     across calls, so these can be matched deterministically.
  3. Score the CALLER's response to each probe, aggregate to one row per call.
  4. Train a small classifier: features -> P(synthetic).
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from src.config import MODELS, TRANSCRIPTS
from src.views.base import View
from src.call import Call


class SemanticView(View):
    name = "semantic"

    def __init__(self):
        self.model = None
        self.vectorizer = None       # e.g. TF-IDF, if you go that route

    # ---- STEP 1: read the pre-computed transcript -------------------------
    def _transcript(self, call: Call) -> list[dict]:
        """Turns for one call, both channels, ordered by start time.

        Produced offline by `python src/transcribe.py`; that run takes ~2 h for
        the full set, so it is never triggered from here.
        """
        path = TRANSCRIPTS / f"{call.anon_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no transcript for {call.anon_id}. Run: python src/transcribe.py"
            )
        return json.loads(path.read_text())["turns"]

    # ---- STEP 2: transcript -> feature vector -----------------------------
    def _features(self, call: Call) -> np.ndarray:
        turns = self._transcript(call)
        # TODO: probe features. Written once we have read ~20 calls and know
        # what the agent's script actually asks. Candidates so far:
        #   - agent states a reference number, then reads it back altered by one
        #     digit -> did the caller object, or confirm it?
        #   - agent offers a false dilemma between two products -> did the caller
        #     pick one, or deny both?
        #   - filler / self-correction rate in the caller's turns
        return np.zeros(16, dtype=np.float32)

    def fit(self, calls: list[Call], y: np.ndarray) -> None:
        X = np.array([self._features(c) for c in calls])
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self._save()

    def proba(self, call: Call) -> float:
        X = self._features(call).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def _save(self):
        with open(MODELS / f"{self.name}.pkl", "wb") as f:
            pickle.dump({"model": self.model, "vectorizer": self.vectorizer}, f)

    def load(self) -> "SemanticView":
        d = pickle.load(open(MODELS / f"{self.name}.pkl", "rb"))
        self.model, self.vectorizer = d["model"], d["vectorizer"]
        return self
