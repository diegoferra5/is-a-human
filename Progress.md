# Progress

Human vs. synthetic caller detection — HackMTY 2026 challenge track.

## Goal

`POST /detect` on stereo 8 kHz telephony audio (ch0 = caller, ch1 = agent) → `{ "is_synthetic": bool, "confidence": float }`.

## Phase 0 — Foundations (done)

### Completed

- [x] Project scaffold — Python package, FastAPI, pytest, CLI entry points
- [x] Audio ingest — in-memory base64 stereo WAV demux, 8 kHz validation, ch0/ch1 split
- [x] Dataset loader — auto-detects `resources/hackmty26-main/` + `resources/audio/`
- [x] Dual Silero VAD — per-channel speech segmentation
- [x] Turn ledger — speech / overlap / silence timeline
- [x] Conversation metrics — talk time, overlap, silence, response latencies
- [x] Eval harness — VAD IoU vs organizer `turns/*.json`, latency stats
- [x] `POST /detect` stub — contract-compliant placeholder verdict
- [x] Test suite — 28 tests (unit + integration on real audio)
- [x] Gitignore + README — dataset folders documented, not committed
- [x] Published branch `Roger` — pushed to origin

### Validation results (val split, 71 calls)

| Metric | Value |
|---|---|
| Mean caller VAD IoU vs organizer turns | 0.811 |
| Mean agent VAD IoU vs organizer turns | 0.941 |
| Mean pipeline latency | 769 ms |
| p95 pipeline latency | 1,041 ms |

### Not done yet

- [ ] Real human vs synthetic classification
- [ ] STT / semantic analysis
- [ ] Signal fusion and calibration
- [ ] Production deployment for judging window

---

## Phase 1 — Exploratory analysis (in progress)

**Hypothesis:** Synthetic callers recover from conversational turbulence (agent interruptions, silence, overlap) more *consistently* than humans. Consistency of timing and turn-taking may separate classes even when raw acoustics are convincing.

**Task:** Run conversational feature extraction on train/val and compare distributions for `human` vs `synthetic`.

```bash
is-a-human-explore --split val
is-a-human-explore --split train
```

Features under study:

- Talk-time ratios (caller vs agent)
- Overlap and silence proportions
- Response latency mean / std / coefficient of variation
- Speech segment counts and utterance length variance
- Overlap event counts

### Findings (val split, n=71: 37 human / 34 synthetic)

Strongest separators by effect size (Cohen's d):

| Feature | Human | Synthetic | d | Direction |
|---|---|---|---|---|
| `agent_talk_ratio` | 0.556 ± 0.049 | 0.440 ± 0.103 | **-1.44** | human talks more (agent channel) |
| `caller_talk_ratio` | 0.184 ± 0.044 | 0.256 ± 0.073 | **+1.19** | synthetic talks more (caller channel) |
| `caller_response_latency_cv` | 0.825 ± 0.136 | 0.703 ± 0.115 | **-0.96** | human latency more variable |
| `agent_response_latency_std_s` | 1.89 ± 0.98 | 3.77 ± 2.78 | **+0.90** | synthetic agent-side timing noisier |
| `agent_response_latency_mean_s` | 2.78 ± 1.00 | 4.80 ± 3.25 | **+0.84** | synthetic slower to respond (agent channel metric) |

**Initial read:**

- Synthetic callers occupy **more caller talk time** (25.6% vs 18.4%) — they may be more verbose or less interrupted.
- Humans have **more agent talk time** — calls may flow differently when a real person is on the line.
- The consistency hypothesis is **partially supported but inverted on caller CV**: humans have *more* variable caller response timing (d = -0.96), not less. Synthetic callers are slightly *more consistent* in when they respond after the agent stops.
- Agent-side latency variance is higher for synthetic calls — may reflect pipeline timing artifacts rather than caller behavior; worth investigating using agent-triggered events only.

**Expanded multi-run analysis (train + val, 200 bootstrap runs):**

Full report: `reports/explore-multi.md`

| Result | Value |
|---|---|
| Split rank correlation | **0.818** (stable across train/val) |
| Stable top features | `caller_rms_mean`, `caller_zcr_std`, `caller_rms_cv` |
| Logistic baseline — train | accuracy 86.9%, F1 0.891, AUC 0.922 |
| Logistic baseline — val | accuracy **84.5%**, F1 0.845, AUC **0.963** |

**Strongest separators (consistent train + val):**

- **Acoustic (unconventional):** synthetic callers have higher mean RMS energy (`caller_rms_mean`, d≈+1.5 to +2.3) but lower variability (`caller_rms_cv`, `caller_zcr_std`, `caller_crest_factor_cv`) — synthetic voices are louder but more uniform.
- **Recovery (agent-turn-aligned):** `agent_aligned_recovery_cv` separates on train (d=-1.32) — humans vary more in how they respond after agent stops.
- **Conversational:** `agent_talk_ratio` still strong on val (d=-1.44).

**Exploration tasks:**

- [x] Re-run on train split to check stability
- [x] Align features to agent turn boundaries (organizer `turns/`)
- [x] Measure recovery after overlap / agent interruption events
- [x] Simple val-set classifier on top features (84.5% val accuracy)

---

## Phase 2 — Detection signal (next)

Pick the strongest separable feature(s) from exploration:

1. **Conversational consistency** (preferred if features separate well)
2. **Acoustic classifier** on caller VAD segments (fallback baseline)
3. **Semantic / trap-question analysis** (only if STT quality holds)

Wire chosen signal into `/detect` with val-set calibration.

---

## Phase 3 — Ship for judging

- [ ] Real verdicts from `/detect`
- [ ] Confidence calibration (ECE, reliability)
- [ ] Latency budget on full calls
- [ ] README approach write-up for judges
- [ ] Hosted endpoint, stable during 15-minute demo

---

## Repo commands

```bash
pip install -e ".[dev]"
pytest                          # all tests
pytest -m "not integration"     # fast unit tests only
is-a-human-eval --split val       # VAD quality benchmark
is-a-human-explore --split val    # human vs synthetic feature comparison
is-a-human-explore --multi        # train+val, bootstrap, logistic baseline
is-a-human-serve                  # local API on :8000
```

## Local data required

```
resources/
  hackmty26-main/   manifest.csv + turns/
  audio/            call_*.wav
```

Both folders are gitignored; must be present locally.
