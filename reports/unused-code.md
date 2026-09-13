# Deleted and unused concepts

Graveyard for code removed from the live package, and ideas we already measured. Read this before adding a new head, fusion style, or acoustic cue. Companion: [`reports/tune/remaining-misses.md`](tune/remaining-misses.md) (val 68/71 = 95.8%).

Kept: `src/is_a_human/` (API, tandem, VAD, extractors, train/eval CLIs) and `tests/`. Deleted only what neither `/detect` nor the test suite imported.

---

## What still runs

```
is-a-human-serve → POST /detect
  demux_base64_telephony
  detect_from_audio (heavy=False, no transcripts)
  hybrid VAD (energy, Silero fallback)
  acoustic + timing + recovery extractors
  TandemModel: 5 acoustic + 3 behavioral + stacked fusion
```

Shipped features in `models/tandem.json`:

- Acoustic: `caller_rms_cv`, `caller_zcr_std`, `caller_crest_factor_cv`, `caller_spectral_flatness_std`, `caller_spectral_centroid_std`
- Behavioral: `caller_response_latency_pos_median_s`, `agent_talk_ratio`, `agent_aligned_recovery_cv`

---

## Deleted files (Max-branch parallel stack)

These lived beside `is_a_human` after the merge. They were never wired into the shipped API or tests. Dependencies they needed (`sklearn`, `xgboost`, `mlx_whisper`, `librosa`, `transformers`) were never in `pyproject.toml`.

| Deleted path | Concept | Why not to rebuild it |
| --- | --- | --- |
| `src/serve.py` | Second FastAPI `/detect` (`audio_b64`, pickle fusion) | Live app is `is_a_human.api.app`. Duplicate contract, behavioral-only. |
| `src/call.py`, `src/config.py` | Disk `Call` object, `dataset/` paths, VAD cache | Live path is in-memory demux. Dataset loader already covers train/eval. |
| `src/fusion.py` | sklearn `LogisticRegression` meta-learner over View `proba`s, K-fold OOF | Same idea as shipped stacked fusion in `detect/tandem.py`, without the disagree override. Concatenated / sklearn fusion did not beat stacked on live VAD. |
| `src/views/base.py` | `View`: `fit` / `proba` / `load` per layer | Abstraction for three parallel owners. We now have two logistic heads + mixer. |
| `src/views/behavioral_view.py` | XGBoost on ~51 timing features, `behavioral.pkl` | Live behavioral logistic already at ~93% val on VAD turns. Dumping the rest of the timing vector overfits collinear wait slices (`pos_mean` / `pos_min` / `pos_max`). |
| `src/views/acoustic_view.py` | Frozen wav2vec2 / XLS-R embedding → logistic. **Never implemented** — returned a 768-dim zero vector | Unmeasured on this 8 kHz Spanish telephony set. Heavy (transformers + resample to 16 kHz). Call-level messiness (`rms_cv`, flatness/centroid std) already does the acoustic job. Do not add a pretrained encoder without Cohen’s d on val then train. |
| `src/views/semantic_view.py` | Transcript probes → logistic. **Stub** — returned zeros | Real probe code now lives in `is_a_human.analysis.semantic` (tests + benchmark only). See semantic below. |
| `src/vad.py` | Energy VAD, 6 dB over 10th-percentile floor | Replaced by `turns/energy_vad.py` (14 dB / 12th percentile) fitted to organizer turns. Hybrid uses that, Silero only if energy is empty. |
| `src/features/build.py`, `src/features/behavioral.py` | CSV table builder; JSON-file wrapper around `extract_timing_features` | Timing extractor is already shared in `analysis/behavioral.py`. |
| `src/train_all.py`, `src/train_behavioral.py`, `src/eval_vad.py` | Train/eval for pickle views | Replaced by `is-a-human-train` / `is-a-human-eval` / `is-a-human-inspect`. |
| `src/transcribe.py` | mlx-whisper large-v3-turbo, Spanish, per-channel 16 kHz, ~2 h for the set | Needed only for semantic. Zero transcripts were on disk at 95.8%. Live Whisper adds seconds of latency. |
| `INTEGRATION.md` | Temporary two-codebase merge brief | Combination shipped. |

---

## Tried and rejected (code still in `is_a_human`, do not re-add to heads)

These extractors still run or still exist for tests/explore. They are **not** in `tandem.json`. Putting them in the shipped heads was already measured.

### Do not add to the acoustic head

- **`caller_rms_mean` (loudness).** Synthetics in this dump are louder at a fixed digital level. Would “save” polite-quiet human `call_6971b2685c1d`. Hidden set may not share that. Ablation: stacked acc held without it.
- **Acoustic means vs CVs.** Mean centroid/flatness/crest/zcr are weaker than the std/CV messiness features we ship. TTS is stable; humans vary.
- **`hf_lf_ratio_*`, intra-silence gaps, segment-length CV.** Extra spectral/gap stats. On 6971, `caller_intra_silence_gap_*` already looks synthetic (VAD glued internal pauses). Unlikely to separate polite humans from bots at call level.

### Do not add to the behavioral head

- **Barge-in, backchannel, overlap counts.** 6971 has ~1 / 1 / 0.14 s vs human means ~9 / 8 / 5 s. They would score that human *more* synthetic.
- **Extra latency slices** (`pos_mean`, `pos_min`, `pos_max`, signed latency, CVs of the same wait). Same floor-switch signal as `pos_median`. Concatenated fusion chewed these and inflated val.
- **`silence_ratio`, recovery mean/count, utterance length stats.** Weaker than `pos_median` + `agent_talk_ratio` + `agent_aligned_recovery_cv`. Pre-tune behavioral used some of these and sat at 79% vs 93% after the three-feature head.

### Do not change fusion this way

- **Concatenated logistic** (all head features in one model). Looks better on some previews with organizer turns and/or `rms_mean`. Live path is VAD. Collinear features. Artifact still saved in `tandem.json` but unused while `fusion_type == "stacked"`.
- **Widen `disagree_unsure` past 0.10** to catch mixer miss `d4518` (margin 0.168). Breaks other val/train humans just outside the band (`bf038`, `9a3d`). Naive max-margin on all val disagreements: save 2 / break 2.
- **Trust the weaker head** to save `e2d538` (behavioral 0.459 vs acoustic 0.879). Coin flip, not a signal.

### Semantic / ASR

- Probe features exist in `analysis/semantic.py` (digit traps, false dilemma, fillers, objections). **Not fields on `CallFeatures`.** Even the latency CLI discards the return value and only times the call.
- Three of five probes are the agent’s script, not the caller.
- 6971’s miss is wait + tidy spectrum, not invented account data.
- Offline val-only transcription is the only honest next semantic experiment. Keep it off `/detect` until effect size and latency are real.

---

## Still in the repo — only leftover with a real hypothesis

`heavy=False` on live and train zeros these. Tests still cover them. Remaining-misses item 2: listen to `e2d538` / measure Cohen’s d **before** shipping.

| Module | Cue | When to touch |
| --- | --- | --- |
| `analysis/micro_variation.py` + `dsp.py` | Pitch jitter/shimmer, formants, HNR, pause entropy, LFCC | If a spectrogram shows vocoder hash that call-level `rms_cv` misses. 6971 has *real* harmonic stacks — jitter may not save it. |
| `analysis/interaction_physics.py` | Breath events, yield decay, echo/bleed correlation | Same bar: val then train effect size, not two-row surgery. |

Do not add a heavy feature that only moves the last two or three val rows. Hidden set is speaker-disjoint.

---

## Three misses at 95.8% (why new code is the wrong first move)

1. **`call_6971b2685c1d` human, both heads wrong.** Polite, quiet, tidy spectrum, 1.75 s waits. The only val call that needs a *new* cue. Loudness, barge-in, overlap, and extra wait slices all point the wrong way.
2. **`call_d4518fcb700a` synthetic, mixer followed timing.** Acoustic already 0.87. Mixer trusts behavioral (weight 2.41 vs 1.65). Feature gap is not the problem.
3. **`call_e2d538297f0c` human, mixer followed acoustics.** Behavioral 0.459 is not a cue to trust.

Ceiling if all three were saved with no collateral: 70/71 ≈ 98.6%. We almost certainly cannot. Hidden-set risk is overfit, not underfit.

Revert shipped heads: `cp models/tandem-before-tune.json models/tandem.json` (pre-tune ~90%).
