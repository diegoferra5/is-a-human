"""semantic view

Signal: what the caller says, and how they handle a script designed to trip them
up. The agent misreads their reference number, interrupts, and asks them to
repeat things back. A person and a language model handle that differently -- but
not in the direction we first guessed. See src/semantic/RESULTS.md.

The short version: the model is the polite, precise, thorough one. It corrects
the misread digit by naming its position ("termina en 06, no 05"), it uses the
agent's name, it performs the full goodbye, and it offers to repeat things
nobody asked it to repeat. The human answers "No." and hangs up.

Pipeline:

    transcript -> features.extract() -> 12 numbers -> logistic regression -> P(synthetic)

Everything up to the regression is deterministic string work; no model runs at
inference time. Transcripts are produced ahead of time by src/transcribe.py and
read from transcripts/<anon_id>.json:

    {"call_id": ..., "turns": [{"channel": 0, "start": 10.5, "end": 20.2, "text": "..."}]}

Turn order and timestamps are kept on purpose: several features are *pairs* --
something the agent said, and how the caller answered right after.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from src.call import Call
from src.config import AGENT_CH, CALLER_CH, MODELS, TRANSCRIPTS
from src.semantic.features import AGENT_DEPENDENT, extract
from src.views.base import View

# The twelve features that separated the classes on train (AUC >= 0.60),
# strongest first. Measured in src/semantic/evaluate.py.
FEATURES = [
    "median_turn_words",        # 0.784
    "one_word_negation",        # 0.776
    "bare_turn_rate",           # 0.762
    "digits_only_turn_rate",    # 0.741
    "closing_words",            # 0.714
    "long_turn_rate",           # 0.702
    "positional_correction",    # 0.684
    "closing_ritual",           # 0.682
    "max_turn_words",           # 0.682
    "calls_agent_by_name",      # 0.672
    "question_turn_rate",       # 0.635
    "offers_to_repeat",         # 0.633
]

# Seven of those twelve count words, and Whisper transcribes clean synthetic
# audio more completely than a human on a noisy line -- so part of their signal
# may be the ASR rather than the speaker. These five never count a word.
# 12 features: AUC 0.898 / 80.8% acc.  These 5: AUC 0.860 / 78.4% acc.
SAFE_FEATURES = [
    "positional_correction",
    "closing_ritual",
    "calls_agent_by_name",
    "question_turn_rate",
    "offers_to_repeat",
]


class SemanticView(View):
    name = "semantic"

    def __init__(self, safe_only: bool = False):
        """safe_only: drop every feature that counts words. Costs ~4 points of
        AUC on train; may well pay for itself on the hidden set, where the
        recording path and ASR conditions are not ours."""
        self.features = SAFE_FEATURES if safe_only else FEATURES
        self.model = None
        self._cache: dict[str, np.ndarray] = {}

    # ---- STEP 1: get the transcript ---------------------------------------
    def _transcript(self, call: Call) -> list[dict]:
        """Turns for one call, both channels, ordered by start time.

        Cache first, Whisper second -- the same shape as Call.turns(). During
        training every call already has a transcript from
        `python src/transcribe.py`, so this is a file read. At serving time
        serve.py hands us a freshly written temp WAV with no transcript, so we
        transcribe it here; that is the slow path and the layer's whole latency
        budget. Results are cached by id, so the fusion's K-fold refits pay for
        a call once.
        """
        path = TRANSCRIPTS / f"{call.anon_id}.json"
        if path.exists():
            return json.loads(path.read_text())["turns"]

        from src.transcribe import MODEL, transcribe_call
        # Only transcribe the channels the chosen features actually read. None
        # of the twelve shipped features look at the agent, so this halves the
        # work on the path where Whisper runs live.
        channels = [CALLER_CH]
        if AGENT_DEPENDENT & set(self.features):
            channels.append(AGENT_CH)
        return transcribe_call(call.audio_path, TRANSCRIPTS,
                               channels=channels, model=MODEL)["turns"]

    # ---- STEP 2: transcript -> feature vector -----------------------------
    def _features(self, call: Call) -> np.ndarray:
        if call.anon_id not in self._cache:
            d = extract(self._transcript(call))
            self._cache[call.anon_id] = np.array(
                [d[k] for k in self.features], dtype=np.float32)
        return self._cache[call.anon_id]

    # ---- STEP 3: fit ------------------------------------------------------
    @staticmethod
    def make_pipeline():
        """The untrained pipeline. Exposed so evaluation scores the same thing
        fit() saves, instead of a look-alike built somewhere else."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Logistic regression on purpose: 282 calls and 12 features is far too
        # little for a tree ensemble, which would memorise speakers instead of
        # the pattern -- and the judging set is speaker-disjoint from both our
        # splits. Twelve weights also mean we can say *why* a call scored high.
        #
        # class_weight="balanced" on purpose too. Without it the model learns
        # our train split's 60/40 class ratio as a standing bias toward
        # "synthetic" -- but val is 47.9% synthetic and the judging set's
        # balance is unknown, so that bias is an assumption we cannot justify.
        # Dropping it costs nothing in AUC (0.898 -> 0.899) and about a point of
        # train accuracy, which was only ever the free advantage of matching the
        # train prior.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))

    def fit(self, calls: list[Call], y: np.ndarray) -> None:
        X = np.array([self._features(c) for c in calls])
        self.model = self.make_pipeline()
        self.model.fit(X, y)
        self._save()

    def proba(self, call: Call) -> float:
        if self.model is None:
            raise RuntimeError("SemanticView is not fitted; call fit() or load()")
        X = self._features(call).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    # ---- explainability ---------------------------------------------------
    def explain(self, call: Call) -> list[tuple[str, float, float]]:
        """(feature, value, contribution) for one call, largest push first.

        Contribution is the standardised value times its weight, so a positive
        number is evidence for synthetic. This is the whole reason for keeping
        the classifier linear.
        """
        if self.model is None:
            raise RuntimeError("SemanticView is not fitted")
        x = self._features(call)
        scaler, clf = self.model[0], self.model[-1]
        z = scaler.transform(x.reshape(1, -1))[0]
        contrib = z * clf.coef_[0]
        return sorted(zip(self.features, x, contrib),
                      key=lambda t: abs(t[2]), reverse=True)

    # ---- persistence ------------------------------------------------------
    def _save(self):
        MODELS.mkdir(exist_ok=True)
        with open(MODELS / f"{self.name}.pkl", "wb") as f:
            pickle.dump({"model": self.model, "features": self.features}, f)

    def load(self) -> "SemanticView":
        with open(MODELS / f"{self.name}.pkl", "rb") as f:
            d = pickle.load(f)
        self.model, self.features = d["model"], d["features"]
        return self
