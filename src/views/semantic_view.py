"""semantic view

Signal: what the caller says and whether it makes sense. The bank agent
deliberately asks nonsense questions, repeats, interrupts. A scripted LLM bot
handles these differently than a confused human.

Plan (pretrained ASR -> text features -> classifier):
  1. Transcribe channel 0 (caller) with Whisper (Spanish). CACHE the transcript
     by anon_id -- Whisper is slow, run it once.  (Transcribe channel 1 too for
     context: what was the caller reacting to?)
  2. Turn transcript(s) into numbers:
       - filler/hesitation words ("este", "eh", "o sea"), self-corrections
       - response relevance: does the caller's turn answer the agent's question?
       - coherence, repetition, over-perfect phrasing, sentence length stats
     (Easy start: TF-IDF or a multilingual sentence-embedding of the transcript.)
  3. Train a small classifier: features -> P(synthetic).

Install: pip install openai-whisper   (or faster-whisper)
Model:   whisper "large-v3" for quality, "small"/"medium" if too slow on M1.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from src.config import MODELS, CACHE
from src.views.base import View
from src.call import Call

_TX_CACHE = CACHE / "transcripts"
_TX_CACHE.mkdir(exist_ok=True)


class SemanticView(View):
    name = "semantic"

    def __init__(self):
        self.model = None
        self.vectorizer = None       # e.g. TF-IDF, if you go that route
        self._asr = None             # lazy-loaded Whisper

    # ---- STEP 1: audio -> transcript (cache by id) ------------------------
    def _transcript(self, call: Call) -> dict:
        cache_file = _TX_CACHE / f"{call.anon_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        # TODO(Person C):
        #   load whisper once (self._asr), transcribe channel 0 (and 1).
        #   tx = {"caller": "...", "agent": "..."}
        tx = {"caller": "", "agent": ""}

        cache_file.write_text(json.dumps(tx))
        return tx

    # ---- STEP 2: transcript -> feature vector -----------------------------
    def _features(self, call: Call) -> np.ndarray:
        tx = self._transcript(call)
        # TODO(Person C): filler counts, relevance, coherence, or a text embedding.
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
