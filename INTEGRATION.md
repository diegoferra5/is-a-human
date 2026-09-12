# Integration brief — merging the two codebases

**Temporary file.** Delete once the combination is done.

This branch (`integration`) is `main` + Roger's branch merged, with both codebases
sitting side by side on purpose so you can see what to keep. Nothing has been
reconciled yet.

---

## What happened

Three people built in parallel without seeing each other's work. Two of them
built the same foundation twice.

| | Max (`src/views/`, `src/features/`) | Roger (`src/is_a_human/`) |
| --- | --- | --- |
| VAD | own, energy-threshold | **Silero** (pretrained, torch) |
| Layer 1 behavioral | latency, turn lengths, overlaps, talk ratio | recovery, interaction physics |
| Layer 2 semantic | stub | — |
| Layer 3 acoustic | stub | **68 features, measured** |
| Fusion | **K-fold stacking over views** | single classifier |
| API | FastAPI `/detect` | FastAPI, own schemas |
| Tests | none | **15 files** |
| Result | none yet | **88.7% val acc, AUC 0.967** |

Layer 2 (semantic) is being built separately from the transcripts and is not in
this merge yet.

---

## What each one is better at

**Roger's is further along.** It runs, it's tested, it has measured results and a
pretrained VAD instead of a hand-rolled threshold. It should probably be the base.

**Max's `View` interface is the better abstraction.** `fit / proba / load` per layer
plus `fusion.py` stacking means three people can work without touching each other's
code, and `serve.py` returns per-view probabilities — which is worth real points at
judging ("which layer caught it?").

Best outcome is likely **Roger's package as the base, Max's View/fusion layer on top.**

---

## Do these first

### 1. Scrub the sponsor and event names — blocking

The repo is public and `CLAUDE.md` forbids both names in committed content. Roger's
branch carries roughly 50 hits across 9 files:

```
tests/test_dataset.py      26      tests/conftest.py            9
README.md                   4      Progress.md                  3
tests/test_paths.py         3      scripts/generate_findings_pdf.py  2
tests/test_eval.py          2      tests/test_explore.py        1
src/is_a_human/dataset/paths.py  1
```

Mostly one pytest fixture named after the event, plus dataset-directory strings. Rename the
fixture and the path constants; nothing functional depends on the wording. I did
not do this myself because renaming across his test suite risks breaking it
silently, and he knows what those fixtures are for.

**Also: never commit `transcripts/`.** The agent speaks the sponsor's bank name in
the opening line of every single call, so every transcript contains it. Already
gitignored — keep it that way, and watch for notebook output cells.

### 2. `caller_rms_mean` is probably a trap — verify before trusting 88.7%

Roger's top feature by a wide margin is **how loud the caller is** (d = +1.50 train,
+2.26 val, rank 1/1 on both). Synthetics are louder.

That is very likely an artifact of how synthetic audio was injected into the
telephony stream at a fixed digital level, while humans held real handsets at
varying distances in varying rooms. It is **not** a property of synthetic speech.

The hidden test set uses unseen callers, unseen engines, and possibly a different
injection setup. That feature could weaken badly or flip sign.

**Test:** retrain with `caller_rms_mean` removed and re-score. If accuracy collapses,
the 88.7% is borrowed and will not survive judging. If it holds, you have a real
model. Either way you need to know before the demo.

The variance features (`caller_rms_cv`, `caller_zcr_std`, `caller_crest_factor_cv`)
are much safer — they measure *messiness*, which is a genuine human/machine
difference rather than a level difference.

### 3. `recovery.py` reads the organizer-provided turns

Its docstring says "from organizer turn segments". `dataset/turns/*.json` does not
exist at judging — we receive audio and nothing else. Either point it at Silero
output or those features cannot ship.

Max's `train_behavioral.py` has the same trap (it can train on provided turns),
and `eval_vad.py` exists precisely to measure that gap. Use it.

### 4. Pick one VAD

Both work. Measured on 30 random calls, Max's energy VAD against the organizers'
turns as ground truth:

| channel | IoU vs ground truth | true speech ratio | VAD says |
| --- | --- | --- | --- |
| 0 caller | **0.90** mean, 0.94 median | 0.29 | 0.33 |
| 1 agent | **0.73** mean | 0.50 | 0.69 |

Caller detection is good. The agent channel is systematically over-detected — turns
get stretched past where they end, which compresses every measured response latency.
Roger's Silero VAD has not been measured this way; run the same comparison and keep
the better one.

A code review flagged this VAD as "broken on noisy calls, 0.84 speech ratio". That
was a bad comparison — it compared one call's VAD output against a *different*
call's ground truth. Zero of 60 channels show VAD above 2x truth. **Do not rewrite
the VAD on the strength of that claim.**

---

## Known bugs in Max's half

Found by review, unfixed in this branch. A separate session is working these on the
`plumbing` branch (a sibling worktree) — check with it before touching.

- `serve.py` — `confidence` returns `abs(p-0.5)*2`, so p=0.05 and p=0.95 both report
  0.9. The brief says confidence is used for tie-breaks and calibration; a symmetric
  certainty cannot be calibration-scored. Should return `p`.
- `fusion.py` — folds are random `StratifiedKFold` over all 353 calls, ignoring that
  train/val are **speaker-disjoint by design**. The same caller lands on both sides,
  inflating the reported score.
- `fusion.py` — the meta-learner is fit on `oof` then scored on that same `oof`, so
  the number printed under "honest out-of-fold metrics" is in-sample for the stacker.
- `train_all.py` and `train_behavioral.py` both write `models/behavioral.pkl`.
  Running the second silently replaces the served model with one trained on provided
  turns.
- `serve.py` — temp WAV leaks on any request that raises; no input validation, so a
  mono upload silently yields all-zero agent features and a confident meaningless verdict.
- `call.py` — the serving path writes a VAD cache keyed by the random tempfile name,
  so it grows forever and never hits.

---

## Layer 2 (semantic) — what is coming

Being built now from Whisper transcripts (`src/transcribe.py`, mlx-whisper
large-v3-turbo, both channels). It plugs into Max's `SemanticView`, already wired to
read `transcripts/<call_id>.json`.

What the agent's script actually does, found by reading transcripts:

1. **Digit trap.** The agent states a reference number, then later reads it back
   **altered by exactly one digit** and asks "¿es correcto?". Confirmed in 3 of the
   first 6 calls: `77120469 -> 77120459`, `66412076 -> 66412075`, `882311457 -> 88231147`.
   A human should catch it; a language model should agree. Detectable with pure string
   work on the agent channel plus the caller's next turn.
2. **False dilemma.** "¿Esto es sobre su cuenta Nómina Plus o sobre su crédito verde?"
   — a forced choice between two products where at least one presumably does not exist.
3. **Announced interruptions.** "Perdón que la interrumpa, [nombre]..." — the agent
   says out loud when it interrupts, so interruption points can be found by phrase
   match in the agent channel rather than only by timing overlap. **Useful to layer 1.**

Caveat: those came from calls that were all `synthetic`. The probes themselves are
solid (the agent script is identical for everyone), but the human-vs-machine
*response* difference is still an untested hypothesis.

---

## Suggested order

1. Scrub the names (blocking, public repo)
2. Drop `caller_rms_mean`, re-score — decide whether 88.7% is real
3. Pick the base package and the VAD, port the other's features in
4. Keep Max's `View` + `fusion` interface so layer 2 can slot in
5. Fix the `plumbing` bugs (or take that branch's work)
