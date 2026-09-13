"""Semantic-head robustness study.

Everything that decides a design is measured on TRAIN with 5-fold out-of-fold
predictions (repeated seeds). Val is scored once per final configuration, for
reporting. Nothing here writes a model.

For each outer fold a full tandem is fit on the fold's training part; the
held-out part gets per-head probabilities. Combiners are then compared on those
held-out probabilities, under scenarios that mask, corrupt or flip the acoustic
and behavioural heads -- the question being whether the semantic head keeps the
model standing when the other two do not.

    python -m is_a_human.eval.robustness --transcripts transcripts_trimmed
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from is_a_human.analysis.explore import _collect_features
from is_a_human.analysis.features import ACOUSTIC_HEAD_FEATURES, BEHAVIORAL_HEAD_FEATURES
from is_a_human.analysis.semantic import SEMANTIC_HEAD_FEATURES
from is_a_human.detect.tandem import (
    GATE_MARGIN,
    _oof_head_probabilities,
    train_tandem_from_rows,
)

CACHE_DIR = Path("cache")


# ---------------------------------------------------------------------------
# small numeric helpers
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    ps = p[order]; i = 0
    while i < len(ps):
        j = i
        while j < len(ps) and ps[j] == ps[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2
        i = j
    n1 = (y == 1).sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {
        "acc": float((pred == y).mean()),
        "auc": auc_score(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "tpr": float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
        "tnr": float((1 - pred[y == 0]).mean()) if (y == 0).any() else float("nan"),
    }


def fit_logit_combiner(X: np.ndarray, y: np.ndarray, l2: float = 0.5, epochs: int = 4000, lr: float = 0.1) -> np.ndarray:
    """Logistic regression on raw logits, no standardisation, so an input of 0
    ("no opinion") contributes exactly nothing. Returns [w..., bias]."""
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        g = Xb.T @ (sigmoid(Xb @ w) - y) / len(y)
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w


# ---------------------------------------------------------------------------
def load_rows(transcripts_dir: str | None, dataset_root: str = "dataset"):
    tag = Path(transcripts_dir).name if transcripts_dir else "none"
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"rows_{tag}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    train = _collect_features("train", dataset_root, heavy=False, vad_backend="hybrid", transcripts_dir=transcripts_dir)
    val = _collect_features("val", dataset_root, heavy=False, vad_backend="hybrid", transcripts_dir=transcripts_dir)
    pickle.dump((train, val), open(cache, "wb"))
    return train, val


def labels_of(rows) -> np.ndarray:
    return np.array([1 if r.label == "synthetic" else 0 for r in rows])


def folds(n: int, k: int, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    for i in range(k):
        va = idx[i::k]
        tr = np.concatenate([idx[j::k] for j in range(k) if j != i])
        yield tr, va


# ---------------------------------------------------------------------------
class FoldModel:
    """One tandem fit on a training set, plus what the combiners need."""

    def __init__(self, train_rows, holdout_rows, *, l2_gate: float = 0.5, hard_weight: float = 1.0,
                 disagree_margin: float = 0.15):
        self.model = train_tandem_from_rows(train_rows, holdout_rows if holdout_rows else train_rows[:2])
        self.margin = self.model.gate_margin
        self.disagree_margin = disagree_margin
        heads = {"p_acoustic": ACOUSTIC_HEAD_FEATURES, "p_behavioral": BEHAVIORAL_HEAD_FEATURES,
                 "p_semantic": SEMANTIC_HEAD_FEATURES}
        # inner OOF of the training part: the same split train_tandem used for its fusion
        inner = _oof_head_probabilities(train_rows, heads)
        self.neutral = {"acoustic": float(self.model.fusion.mean[0]), "behavioral": float(self.model.fusion.mean[1])}
        p_fast_inner = self._fast(inner[:, 0], inner[:, 1])
        y_inner = labels_of(train_rows)
        gated = np.abs(p_fast_inner - 0.5) < self.margin
        if hard_weight != 1.0:
            # hard-example weighting: the semantic head is only consulted inside
            # the gate, so let it care more about the rows that land there
            from is_a_human.analysis.classifier import train_logistic_regression
            w = np.where(gated, hard_weight, 1.0)
            self.model.semantic = train_logistic_regression(train_rows, SEMANTIC_HEAD_FEATURES, sample_weight=w)
        X = np.c_[logit(p_fast_inner[gated]), logit(inner[gated, 2])]
        self.w_gate = fit_logit_combiner(X, y_inner[gated], l2=l2_gate) if gated.sum() >= 8 else np.array([0.5, 0.5, 0.0])
        self.n_gated_inner = int(gated.sum())

    # head probabilities for held-out rows
    def heads(self, rows):
        return {
            "acoustic": np.array([self.model.acoustic.predict_one(r) for r in rows]),
            "behavioral": np.array([self.model.behavioral.predict_one(r) for r in rows]),
            "semantic": np.array([self.model.semantic.predict_one(r) for r in rows]),
        }

    def _fast(self, p_ac, p_beh):
        rows = [SimpleNamespace(p_acoustic=float(a), p_behavioral=float(b)) for a, b in zip(p_ac, p_beh)]
        return self.model.fusion.predict_proba_rows(rows)

    def _full3(self, p_ac, p_beh, p_sem):
        rows = [SimpleNamespace(p_acoustic=float(a), p_behavioral=float(b), p_semantic=float(s))
                for a, b, s in zip(p_ac, p_beh, p_sem)]
        return self.model.fusion3.predict_proba_rows(rows)

    def combine(self, h: dict, forced_gate: np.ndarray | None = None,
                masked: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        p_ac, p_beh, p_sem = h["acoustic"], h["behavioral"], h["semantic"]
        n = len(p_ac)
        masked = masked or {"acoustic": np.zeros(n, bool), "behavioral": np.zeros(n, bool)}
        p_fast = self._fast(p_ac, p_beh)
        inside = np.abs(p_fast - 0.5) < self.margin
        if forced_gate is not None:
            inside = inside | forced_gate
        # disagreement trigger: the two fast heads on opposite sides, both by a margin
        dm = self.disagree_margin
        disagree = ((p_ac - 0.5) * (p_beh - 0.5) < 0) & (np.abs(p_ac - 0.5) > dm) & (np.abs(p_beh - 0.5) > dm)
        disagree &= ~masked["acoustic"] & ~masked["behavioral"]
        inside_d = inside | disagree
        out = {"fast": p_fast, "semantic_only": p_sem}
        self.last_gated = {"unsure": float(inside.mean()), "unsure_or_disagree": float(inside_d.mean())}
        # 3-way average of available heads' log-odds; masked heads drop out
        la = np.where(masked["acoustic"], 0.0, logit(p_ac))
        lb = np.where(masked["behavioral"], 0.0, logit(p_beh))
        cnt = 1.0 + (~masked["acoustic"]) + (~masked["behavioral"])
        three = sigmoid((la + lb + logit(p_sem)) / cnt)
        out["dgate_3way_avg"] = np.where(inside_d, three, p_fast)
        out["dgate_logit_avg"] = np.where(inside_d, sigmoid((logit(p_fast) + logit(p_sem)) / 2), p_fast)
        # current shipped design: fusion3 (fit on all rows) inside the gate
        p3 = self._full3(p_ac, p_beh, p_sem)
        out["gated_fusion3_all"] = np.where(inside, p3, p_fast)
        out["dgate_fusion3_all"] = np.where(inside_d, p3, p_fast)
        out["always_fusion3_all"] = p3
        # parameter-free: average the log-odds of fast and semantic inside the gate
        avg = sigmoid((logit(p_fast) + logit(p_sem)) / 2)
        out["gated_logit_avg"] = np.where(inside, avg, p_fast)
        # fitted on the gated region of the inner OOF, on logits (masked head = 0)
        w = self.w_gate
        fit = sigmoid(w[0] * logit(p_fast) + w[1] * logit(p_sem) + w[2])
        out["gated_fit"] = np.where(inside, fit, p_fast)
        return out


# scenarios act on held-out head probabilities
def apply_scenario(h: dict, name: str, neutral: dict, rng: np.random.Generator):
    h = {k: v.copy() for k, v in h.items()}
    forced = None
    n = len(h["acoustic"])
    masked = {"acoustic": np.zeros(n, bool), "behavioral": np.zeros(n, bool)}

    def noisy(p, sd=1.5):
        return sigmoid(logit(p) + rng.normal(0, sd, size=len(p)))

    if name == "clean":
        pass
    elif name == "acoustic_masked":
        h["acoustic"][:] = neutral["acoustic"]; forced = np.ones(n, bool); masked["acoustic"][:] = True
    elif name == "behavioral_masked":
        h["behavioral"][:] = neutral["behavioral"]; forced = np.ones(n, bool); masked["behavioral"][:] = True
    elif name == "both_masked":
        h["acoustic"][:] = neutral["acoustic"]; h["behavioral"][:] = neutral["behavioral"]; forced = np.ones(n, bool)
        masked["acoustic"][:] = True; masked["behavioral"][:] = True
    elif name == "acoustic_noisy":
        h["acoustic"] = noisy(h["acoustic"])
    elif name == "behavioral_noisy":
        h["behavioral"] = noisy(h["behavioral"])
    elif name == "both_noisy":
        h["acoustic"] = noisy(h["acoustic"]); h["behavioral"] = noisy(h["behavioral"])
    elif name == "acoustic_flipped":
        h["acoustic"] = 1 - h["acoustic"]
    elif name == "behavioral_flipped":
        h["behavioral"] = 1 - h["behavioral"]
    elif name == "semantic_masked":
        h["semantic"][:] = 0.5
    else:
        raise ValueError(name)
    return h, forced, masked


SCENARIOS = ("clean", "acoustic_masked", "behavioral_masked", "both_masked",
             "acoustic_noisy", "behavioral_noisy", "both_noisy",
             "acoustic_flipped", "behavioral_flipped", "semantic_masked")
COMBINERS = ("fast", "gated_fusion3_all", "gated_logit_avg", "gated_fit",
             "dgate_logit_avg", "dgate_3way_avg", "dgate_fusion3_all", "always_fusion3_all", "semantic_only")


def train_oof_study(train_rows, seeds=(0, 1, 2), k=5, l2_gate=0.5, hard_weight=1.0, disagree_margin=0.15) -> dict:
    y = labels_of(train_rows)
    acc: dict[str, dict[str, list]] = {s: {c: [] for c in COMBINERS} for s in SCENARIOS}
    aucs: dict[str, dict[str, list]] = {s: {c: [] for c in COMBINERS} for s in SCENARIOS}
    gated_frac = []
    dgated_frac = []
    n_gated_inner = []
    for seed in seeds:
        oof = {s: {c: np.zeros(len(y)) for c in COMBINERS} for s in SCENARIOS}
        rng = np.random.default_rng(100 + seed)
        for tr, va in folds(len(y), k, seed):
            fm = FoldModel([train_rows[i] for i in tr], [train_rows[i] for i in va], l2_gate=l2_gate,
                           hard_weight=hard_weight, disagree_margin=disagree_margin)
            n_gated_inner.append(fm.n_gated_inner)
            h0 = fm.heads([train_rows[i] for i in va])
            p_fast = fm._fast(h0["acoustic"], h0["behavioral"])
            gated_frac.append(float((np.abs(p_fast - 0.5) < fm.margin).mean()))
            for s in SCENARIOS:
                h, forced, masked = apply_scenario(h0, s, fm.neutral, rng)
                for c, p in fm.combine(h, forced, masked).items():
                    oof[s][c][va] = p
                if s == "clean":
                    dgated_frac.append(fm.last_gated["unsure_or_disagree"])
        for s in SCENARIOS:
            for c in COMBINERS:
                m = metrics(y, oof[s][c]); acc[s][c].append(m["acc"]); aucs[s][c].append(m["auc"])
    return {
        "n": int(len(y)), "seeds": list(seeds),
        "gated_fraction": float(np.mean(gated_frac)),
        "dgated_fraction": float(np.mean(dgated_frac)),
        "gated_rows_for_fit": float(np.mean(n_gated_inner)),
        "acc": {s: {c: float(np.mean(v)) for c, v in d.items()} for s, d in acc.items()},
        "acc_sd": {s: {c: float(np.std(v)) for c, v in d.items()} for s, d in acc.items()},
        "auc": {s: {c: float(np.mean(v)) for c, v in d.items()} for s, d in aucs.items()},
    }


def val_study(train_rows, val_rows, l2_gate=0.5, hard_weight=1.0, disagree_margin=0.15) -> dict:
    fm = FoldModel(train_rows, val_rows, l2_gate=l2_gate, hard_weight=hard_weight, disagree_margin=disagree_margin)
    y = labels_of(val_rows); h0 = fm.heads(val_rows); rng = np.random.default_rng(7)
    out = {}
    for s in SCENARIOS:
        h, forced, masked = apply_scenario(h0, s, fm.neutral, rng)
        out[s] = {c: metrics(y, p) for c, p in fm.combine(h, forced, masked).items()}
        if s == "clean":
            out["_gated_fraction"] = fm.last_gated["unsure"]
            out["_dgated_fraction"] = fm.last_gated["unsure_or_disagree"]
    return out


def render(train: dict, val: dict | None, title: str) -> str:
    L = [f"# {title}", ""]
    L += [f"Train OOF: 5-fold × seeds {train['seeds']}, n={train['n']}. Gate margin {GATE_MARGIN}: "
          f"{100*train['gated_fraction']:.1f}% of calls gated by unsure, "
          f"{100*train['dgated_fraction']:.1f}% by unsure-or-disagree; "
          f"gate combiner fit on ~{train['gated_rows_for_fit']:.0f} inner rows.", ""]
    L += ["## Train OOF accuracy (mean over seeds)", "", "| scenario | " + " | ".join(COMBINERS) + " |",
          "|---|" + "---|" * len(COMBINERS)]
    for s in SCENARIOS:
        L.append(f"| {s} | " + " | ".join(f"{100*train['acc'][s][c]:.1f}" for c in COMBINERS) + " |")
    L += ["", "## Train OOF AUC", "", "| scenario | " + " | ".join(COMBINERS) + " |", "|---|" + "---|" * len(COMBINERS)]
    for s in SCENARIOS:
        L.append(f"| {s} | " + " | ".join(f"{train['auc'][s][c]:.3f}" for c in COMBINERS) + " |")
    if val:
        L += ["", f"## Val (71 unseen voices) — scored once. Gated: unsure {100*val['_gated_fraction']:.1f}%, "
              f"unsure-or-disagree {100*val.get('_dgated_fraction', 0):.1f}%", "",
              "| scenario | " + " | ".join(f"{c} acc / auc" for c in COMBINERS) + " |", "|---|" + "---|" * len(COMBINERS)]
        for s in SCENARIOS:
            L.append(f"| {s} | " + " | ".join(f"{100*val[s][c]['acc']:.1f} / {val[s][c]['auc']:.3f}" for c in COMBINERS) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="transcripts")
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--l2-gate", type=float, default=0.5)
    ap.add_argument("--val", action="store_true", help="also score val once")
    ap.add_argument("--hard-weight", type=float, default=1.0, help="extra weight on gated rows when fitting the semantic head")
    ap.add_argument("--disagree-margin", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    train_rows, val_rows = load_rows(args.transcripts, args.dataset_root)
    tr = train_oof_study(train_rows, seeds=tuple(range(args.seeds)), l2_gate=args.l2_gate,
                         hard_weight=args.hard_weight, disagree_margin=args.disagree_margin)
    va = val_study(train_rows, val_rows, l2_gate=args.l2_gate, hard_weight=args.hard_weight,
                   disagree_margin=args.disagree_margin) if args.val else None
    title = (f"Semantic robustness — transcripts={args.transcripts}, l2_gate={args.l2_gate}, "
             f"hard_weight={args.hard_weight}, disagree_margin={args.disagree_margin}")
    md = render(tr, va, title)
    out = args.out or Path("reports/robustness") / f"{Path(args.transcripts).name}_l2{args.l2_gate}_hw{args.hard_weight}_dm{args.disagree_margin}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"train": tr, "val": va}, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
