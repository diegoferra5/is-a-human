"""acoustic view

Signal: HOW the caller's voice SOUNDS. Synthetic voices leave subtle artifacts
(vocoder buzz, unnatural smoothness) even at 8 kHz.

for frozen pretrained model as a feature extractor -> small classifier:
  1. Take channel 0 (caller). Resample 8 kHz -> 16 kHz (what the model expects).
  2. Run it through a FROZEN wav2vec2 / XLS-R model -> one embedding vector. this is the pretrained model
     (XLS-R is multilingual -> good for Spanish.)  CACHE it by anon_id.
  3. Train logistic regression / small MLP: embedding -> P(synthetic).

Tip: also try cheap prosody features (pitch jitter/shimmer) normalized against
channel 1 (the agent = known human on the same line) as extra inputs.

make sure to pip install soundfile librosa   (torch + transformers already present)
Model:   transformers Wav2Vec2Model, e.g. "facebook/wav2vec2-xls-r-300m"
"""
from __future__ import annotations

import pickle

import numpy as np

from src.config import MODELS, CACHE
from src.views.base import View
from src.call import Call

_EMB_CACHE = CACHE / "acoustic_emb"
_EMB_CACHE.mkdir(exist_ok=True)


class AcousticView(View):
    name = "acoustic"

    def __init__(self):
        self.model = None            # the small classifier head
        self._encoder = None         # lazy-loaded frozen wav2vec2

    # audio -> embedding (cache by id) 
    def _embedding(self, call: Call) -> np.ndarray:
        cache_file = _EMB_CACHE / f"{call.anon_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        # TODO:
        #   x, sr = call.caller_audio()           # channel 0, 8 kHz
        #   x16 = resample(x, sr, 16000)          # librosa.resample
        #   emb = frozen_wav2vec2(x16).mean(axis=time)   # pooled embedding
        # For now return zeros so the pipeline runs end-to-end.
        emb = np.zeros(768, dtype=np.float32)

        np.save(cache_file, emb)
        return emb

    # train the small head
    def fit(self, calls: list[Call], y: np.ndarray) -> None:
        X = np.array([self._embedding(c) for c in calls])
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        # TODO(Person B): once embeddings are real, this line already works.
        self.model.fit(X, y)
        self._save()

    def proba(self, call: Call) -> float:
        X = self._embedding(call).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def _save(self):
        with open(MODELS / f"{self.name}.pkl", "wb") as f:
            pickle.dump({"model": self.model}, f)

    def load(self) -> "AcousticView":
        d = pickle.load(open(MODELS / f"{self.name}.pkl", "rb"))
        self.model = d["model"]
        return self
