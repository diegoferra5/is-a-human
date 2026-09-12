# Progress

Human vs. synthetic caller detection — the hackathon challenge track.

## Goal

`POST /detect` on stereo 8 kHz telephony audio (ch0 = caller, ch1 = agent) → `{ "is_synthetic": bool, "confidence": float }`.

---

## Current state (important)

**We have the metrics. We do not have a live tester that uses them yet.**

| Capability | Status | Where |
|---|---|---|
| 48 features per call (18 conversational + 17 acoustic + 13 recovery) | Done | `is-a-human-explore`, offline only |
| Logistic baseline (~84.5% val accuracy) | Done | Trained inside `--multi`, **not saved or loaded** |
| Batch analysis on full dataset | Done | CLI + `reports/` |
| PDF reports (technical + plain language) | Done | `reports/is-a-human-findings.pdf`, `...-simple.pdf` |
| Live API `POST /detect` | Stub only | Always returns `is_synthetic=false`, `confidence=0.5` |
| Saved model artifact | Missing | Classifier discarded after each explore run |
| Per-upload metrics endpoint | Missing | No `/analyze` route |
| Dev visual UI | **Not built** | Planned — see below |

What `/detect` does today: accepts audio, runs VAD internally, **discards the result**, returns a placeholder. The discovery metrics and classifier exist only in offline batch mode.

### Critical gap: offline vs live feature path

Batch exploration and the logistic baseline do **not** use the same turn source end-to-end:

| Feature group | Turn source today | Live `/detect` ready? |
|---|---|---|
| Conversational (talk ratios, latencies, segment counts) | VAD ledger from `process_call()` | Yes |
| Acoustic (`caller_rms_*`, `caller_zcr_*`, spectral, crest) | **Organizer** `turns/*.json` segments | **No** — must switch to VAD segments |
| Recovery (agent-aligned, barge-in, overlap) | **Organizer** `turns/*.json` segments | **No** — optional for v1 (not in top-3 model) |

The shipped classifier uses only the three stable acoustic separators. Those were computed on organizer turn boundaries during exploration. Before live detection works, `extract_acoustic_features()` must accept VAD-derived caller segments (from the turn ledger) and we should re-measure val accuracy on that path — caller VAD IoU is 0.811, so some drift is expected.

Recovery features can stay organizer-only for dev/dataset mode initially; they are not required for the v1 three-feature model.

---

## Architecture

```
src/is_a_human/
  audio/          demux base64/WAV, 8 kHz stereo validation
  dataset/        manifest + turns loader, path auto-detection
  turns/          Silero VAD, turn ledger, conversation metrics
  pipeline.py     process_call() — VAD → ledger → metrics
  eval/           offline VAD IoU benchmark (is-a-human-eval)
  analysis/       feature extraction, explore, logistic classifier
  api/            FastAPI /health + /detect (stub verdict)
scripts/          PDF report generators (matplotlib + reportlab)
reports/          explore-multi.md, findings PDFs, _charts/
tests/            33 tests across 11 modules
```

**Classifier (offline only):** manual logistic regression in `analysis/classifier.py` — trains on the intersection of stable top features from train and val (`caller_rms_mean`, `caller_zcr_std`, `caller_rms_cv`), z-score normalized, 2500 epochs, L2=0.01. No sklearn; weights live in `TrainedLogistic` dataclass only.

**Model artifact:** `.gitignore` excludes `*.joblib` / `*.pkl`. Planned path: `models/baseline.json` (weights + mean/std + feature names) loaded at API startup.

---

## Phase 0 — Foundations (done)

### Completed

- [x] Project scaffold — Python package, FastAPI, pytest, CLI entry points
- [x] Audio ingest — in-memory base64 stereo WAV demux, 8 kHz validation, ch0/ch1 split
- [x] Dataset loader — auto-detects `resources/challenge-dataset/` + `resources/audio/`
- [x] Dual Silero VAD — per-channel speech segmentation
- [x] Turn ledger — speech / overlap / silence timeline
- [x] Conversation metrics — talk time, overlap, silence, response latencies
- [x] Eval harness — VAD IoU vs organizer `turns/*.json`, latency stats
- [x] `POST /detect` stub — contract-compliant placeholder verdict
- [x] Test suite — 33 tests (unit + integration on real audio)
- [x] Gitignore + README — dataset folders documented, not committed
- [x] Published branch `Roger` — pushed to origin

### Validation results (val split, 71 calls)

| Metric | Value |
|---|---|
| Mean caller VAD IoU vs organizer turns | 0.811 |
| Mean agent VAD IoU vs organizer turns | 0.941 |
| Mean pipeline latency | 769 ms |
| p95 pipeline latency | 1,041 ms |

---

## Phase 1 — Exploratory analysis (done)

**Hypothesis:** Synthetic callers recover from conversational turbulence more *consistently* than humans.

**Outcome:** Acoustic uniformity on the caller channel is the stronger signal. Synthetic voices are louder but more uniform; humans are messier. Conversational consistency hypothesis was partially inverted (humans vary more, not less).

### Reports

- Markdown: `reports/explore-multi.md`
- Technical PDF: `reports/is-a-human-findings.pdf`
- Plain-language PDF: `reports/is-a-human-findings-simple.pdf`

### Key results (train + val, 200 bootstrap runs)

| Result | Value |
|---|---|
| Split rank correlation | **0.818** |
| Stable top features | `caller_rms_mean`, `caller_zcr_std`, `caller_rms_cv` |
| Logistic baseline — train | accuracy 86.9%, F1 0.891, AUC 0.922 |
| Logistic baseline — val | accuracy **84.5%**, F1 0.845, AUC **0.963** |

### What separates human vs synthetic (interpretation)

| Signal | Pattern | Cohen's d (train) |
|---|---|---|
| `caller_rms_mean` | Synthetic callers **louder** on average | +1.50 |
| `caller_zcr_std` | Humans **more variable** zero-crossing rate | −1.37 |
| `caller_rms_cv` | Humans **more variable** loudness | −1.37 |
| `agent_talk_ratio` | Humans leave **more agent talk time** (val only) | −1.44 |

Synthetic voices look like a consistent, amplified pipeline; humans are quieter and noisier turn-to-turn. The original “recovery consistency” hypothesis held on train (`agent_aligned_recovery_cv`, d=−1.32) but acoustic uniformity generalizes better across splits (rank correlation 0.818).

### Exploration tasks

- [x] Re-run on train split to check stability
- [x] Align features to agent turn boundaries (organizer `turns/`)
- [x] Measure recovery after overlap / agent interruption events
- [x] Simple val-set classifier on top features
- [x] Expanded acoustic + recovery feature set (48 features)
- [x] Multi-run bootstrap analysis
- [x] PDF findings reports

---

## Phase 1.5 — Dev visual UI (planned, next)

A **data-rich local web UI** for developer testing — not the production judging endpoint, but a dashboard to upload a call and inspect everything the pipeline produces.

### Purpose

- Test single calls interactively without running batch CLIs
- Validate features, verdict, and latency before wiring `/detect`
- Compare uploaded audio against train/val reference distributions
- Debug misclassifications during model iteration

### Planned UI (single-page dev dashboard)

**Upload panel**
- Drag-and-drop or file picker for stereo WAV (8 kHz)
- Optional: pick a call from local dataset by `anon_id` (if `resources/` present)
- Show channel mapping reminder (ch0 = caller, ch1 = agent)

**Verdict panel**
- `is_synthetic` + `confidence` from live classifier (once wired)
- Processing latency (ms)
- Ground-truth label if loaded from dataset (for dev comparison)

**Metrics panel (data-rich)**
- All 48 extracted features in grouped tables:
  - Conversational (talk ratios, overlap, latencies)
  - Acoustic (RMS, ZCR, spectral, crest factor)
  - Recovery (agent-aligned, barge-in, backchannels)
- Highlight top stable separators vs uploaded values
- Show percentile vs train distribution (human / synthetic)

**Timeline panel**
- Dual-channel turn ledger visualization (speech / overlap / silence)
- Caller vs agent speech segments on a shared timeline
- VAD segments overlaid (ours vs organizer turns when available)

**Charts panel**
- Radar or bar chart: uploaded call vs human mean vs synthetic mean
- Feature contribution / classifier weights for the verdict

**Tech approach (proposed)**
- FastAPI backend: new `POST /analyze` (full feature JSON) + wire classifier into `/detect`
- Static or lightweight frontend served from `src/is_a_human/ui/` (HTML + JS, no separate build step for v1)
- Train and persist model to `models/baseline.joblib` (or JSON weights) at startup
- Live calls use VAD-derived turns (no organizer `turns/` required)

### Dev UI tasks

- [ ] Persist trained logistic model artifact from train split
- [ ] `POST /analyze` — upload WAV, return features + ledger + metrics + verdict
- [ ] Wire saved model into `/detect` (real verdicts)
- [ ] Build dev dashboard UI (upload, verdict, metrics tables, timeline, charts)
- [ ] Dataset picker — load known calls with label for side-by-side dev testing
- [ ] Reference distribution overlays (human/synthetic means from train)

---

## Phase 2 — Live detection (next)

**Chosen signal:** acoustic features (`caller_rms_mean`, `caller_zcr_std`, `caller_rms_cv`) based on Phase 1 findings. Conversational/recovery features as optional secondary inputs later.

### Ordered implementation (recommended)

1. **VAD-backed acoustic extraction** — refactor `acoustic.py` / `features.py` to derive caller segments from the turn ledger when organizer turns are absent; add tests comparing VAD vs organizer paths on val split.
2. **Train + persist model** — CLI or startup hook: `is-a-human-explore --multi` logic → save `TrainedLogistic` to `models/baseline.json`; commit reference stats (human/synthetic means per feature) for dev UI overlays.
3. **Wire `/detect`** — in `api/app.py`: after `process_call()`, extract 3 features, load model, return `is_synthetic=(prob>=0.5)`, `confidence=prob`.
4. **`POST /analyze`** — same pipeline, return full feature dict + ledger events + latency ms (dev endpoint, not for judges).
5. **Re-benchmark val** — accuracy/AUC on VAD-backed features; target within ~5 pts of 84.5% offline baseline.
6. **Dev UI** (Phase 1.5) — build on `/analyze` once backend is stable.

### Tasks

- [ ] VAD-backed acoustic feature path (blocker for everything below)
- [ ] Save model artifact + feature scaler from train split
- [ ] Feature extraction on live upload (VAD-derived turns, no precomputed JSON)
- [ ] Real `/detect` verdicts with calibrated confidence
- [ ] Dev UI (Phase 1.5) for interactive testing

Not in scope yet:
- [ ] STT / semantic analysis
- [ ] Signal fusion beyond logistic baseline
- [ ] Production deployment for judging window

---

## Phase 3 — Ship for judging

- [ ] Confidence calibration (ECE, reliability)
- [ ] Latency budget validated on full calls via UI + API
- [ ] README approach write-up for judges
- [ ] Hosted endpoint, stable during 15-minute demo

---

## Test coverage (33 passing)

| Module | Tests | Notes |
|---|---|---|
| `audio/demux` | unit | base64, stereo, sample rate validation |
| `dataset/loader` | unit | manifest, turns, path resolution |
| `turns/` | unit | ledger, VAD integration |
| `pipeline` | unit + integration | real audio when dataset present |
| `eval/` | unit | harness metrics |
| `analysis/` | unit | explore, classifier, recovery |
| `api/` | unit + integration | `/health`, `/detect` contract |

Run with venv active: `source .venv/bin/activate && pytest`

---

## Repo commands

```bash
pip install -e ".[dev]"
pytest                          # all tests
pytest -m "not integration"     # fast unit tests only
is-a-human-eval --split val       # VAD quality benchmark
is-a-human-explore --split val    # human vs synthetic feature comparison
is-a-human-explore --multi        # train+val, bootstrap, logistic baseline
is-a-human-explore --multi --output reports/explore-multi.md  # refresh report
is-a-human-serve                  # local API on :8000 (stub /detect today)
python scripts/generate_findings_pdf.py         # technical PDF
python scripts/generate_findings_pdf_simple.py  # plain-language PDF
```

PDF scripts require `matplotlib` and `reportlab` (installed in local venv; not yet in `pyproject.toml` optional deps).

## Local data required

```
resources/
  challenge-dataset/   manifest.csv + turns/
  audio/            call_*.wav
```

Both folders are gitignored; must be present locally.

---

## Changelog

| Date | Milestone |
|---|---|
| Phase 0 | Pipeline, eval harness, API stub, 33 tests |
| Phase 1 | 48-feature explore, logistic baseline 84.5% val, PDF reports |
| Next | VAD-backed acoustic path → persist model → wire `/detect` → dev UI |

*Last updated: 2026-09-11 — branch `Roger`, commit `Findings`.*
