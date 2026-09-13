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
from src.config import AGENT_CH, CACHE, CALLER_CH, MODELS, TRANSCRIPTS
from src.semantic.features import AGENT_DEPENDENT, extract
from src.views.base import View

# The twelve features that separated the classes on train (AUC >= 0.60),
# strongest first. Measured in src/semantic/evaluate.py.
FEATURES = [
    "speech_rate",              # 0.885 alone -- the layer's main signal
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
    # offers_to_repeat was here at 0.633. A review showed the score came
    # entirely from the phrase "con eso" ("no, con eso esta bien" -- declining
    # help). With that removed it scores 0.537, i.e. nothing. Dropped.
]

# A smaller variant. It was originally the features that "never count a word",
# on the worry that Whisper heard the two groups differently. A review killed
# that worry: the agent is the same TTS voice on every call and its words per
# second are identical across groups, so the ASR is not the difference. The
# list is kept as a compact alternative -- speech rate plus the four
# behaviour features -- not as a safety net.
SAFE_FEATURES = [
    "speech_rate",
    "positional_correction",
    "closing_ritual",
    "calls_agent_by_name",
    "question_turn_rate",
]


class SemanticView(View):
    name = "semantic"

    def __init__(self, safe_only: bool = False):
        """safe_only: the compact 5-feature variant (see SAFE_FEATURES).
        Saves to and loads from models/semantic_safe.pkl."""
        self.features = SAFE_FEATURES if safe_only else FEATURES
        # Separate files so `train --safe` cannot silently replace the full
        # model. The view *name* stays "semantic" -- it is the fusion's key.
        self.pkl = MODELS / ("semantic_safe.pkl" if safe_only else "semantic.pkl")
        self.model = None
        # Univariate direction of each feature on the training data (+1 if the
        # synthetic mean is higher). Kept so explain() can refuse to show a
        # weight whose sign the marginal statistics contradict.
        self.direction: dict[str, int] = {}
        # Keyed by (anon_id, feature set): the same call under a different
        # feature list is a different vector, and load() may swap the list.
        self._cache: dict[tuple, np.ndarray] = {}

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
        # out_dir=None: never write a live request's transcript into the
        # training corpus. It would be caller-channel-only and would poison
        # any later training run that globs transcripts/.
        return transcribe_call(call.audio_path, None,
                               channels=channels, model=MODEL)["turns"]

    # ---- STEP 2: transcript -> feature vector -----------------------------
    @staticmethod
    def caller_speech_seconds(call: Call) -> float:
        """Seconds the caller actually spoke, by the ENERGY VAD, cached by id.

        Not Call.turns(): that is Silero, and Silero at 8 kHz under-detects
        human phone speech by about 25% while over-counting the TTS caller
        (measured against dataset/turns: human 0.75x truth, synthetic 1.00x,
        and lowering the threshold widens the gap). A class-biased denominator
        would turn speech_rate into a VAD artefact. The energy VAD over-counts
        both classes by the same ~1.25x, so the ratio is honest: AUC 0.833
        against 0.760 with Silero and 0.885 with the dataset's own turns.
        """
        from src.vad import extract_turns
        cache = CACHE / "vad_turns_energy"
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"{call.anon_id}.json"
        if f.exists() and f.stat().st_size > 0:
            turns = json.loads(f.read_text())["turns"]
        else:
            d = extract_turns(call.audio_path, backend="energy")
            f.write_text(json.dumps(d))
            turns = d["turns"]
        return sum(t["end"] - t["start"] for t in turns if t["channel"] == CALLER_CH)

    def _features(self, call: Call) -> np.ndarray:
        key = (call.anon_id, tuple(self.features))
        if key not in self._cache:
            caller_s = self.caller_speech_seconds(call)
            d = extract(self._transcript(call), caller_speech_s=caller_s)
            self._cache[key] = np.array([d[k] for k in self.features], dtype=np.float32)
        return self._cache[key]

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
        # Balanced vs unweighted: AUC 0.899 vs 0.898, i.e. nothing; it gives
        # up about a point of train accuracy, which was only ever the free
        # advantage of matching the train prior.
        #
        # C=0.1 (default is 1.0): several features are correlated (|r| up to
        # 0.70), and at C=1.0 three of the small weights had undetermined sign
        # under bootstrap. Stronger shrinkage steadies them; grouped-CV AUC is
        # unchanged to +0.006.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1))

    def fit(self, calls: list[Call], y: np.ndarray) -> None:
        X = np.array([self._features(c) for c in calls])
        y = np.asarray(y)
        self.model = self.make_pipeline()
        self.model.fit(X, y)
        diff = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
        self.direction = {f: int(np.sign(d)) for f, d in zip(self.features, diff)}
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

        A feature is left out when the sign of its fitted weight contradicts
        its univariate direction on the training data. That happens to small
        weights on correlated features (a suppressor effect), and showing it
        would tell a reader "many short turns -> synthetic" while the marginal
        table says the opposite. The prediction still uses every feature;
        only the explanation is filtered.
        """
        if self.model is None:
            raise RuntimeError("SemanticView is not fitted")
        x = self._features(call)
        scaler, clf = self.model[0], self.model[-1]
        z = scaler.transform(x.reshape(1, -1))[0]
        contrib = z * clf.coef_[0]
        rows = [(f, float(v), float(c))
                for f, v, c, w in zip(self.features, x, contrib, clf.coef_[0])
                if self.direction.get(f, 0) == 0 or np.sign(w) == self.direction[f]]
        return sorted(rows, key=lambda t: abs(t[2]), reverse=True)

    # ---- persistence ------------------------------------------------------
    def _save(self):
        MODELS.mkdir(exist_ok=True)
        with open(self.pkl, "wb") as f:
            pickle.dump({"model": self.model, "features": self.features,
                         "direction": self.direction}, f)

    def load(self) -> "SemanticView":
        with open(self.pkl, "rb") as f:
            d = pickle.load(f)
        self.model, self.features = d["model"], d["features"]
        self.direction = d.get("direction", {})
        self._cache.clear()          # vectors built under another feature list are stale
        return self
