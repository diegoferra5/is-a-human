# Remaining misses — what not to do

Shipped detector: stacked acoustic + behavioural, hybrid VAD, no `caller_rms_mean`, unsure-disagreement override (`|p_fused − 0.5| < 0.10` → sharper head).

Val **68/71 = 95.8%** · balanced **95.8%** (TPR 97.1% / TNR 94.6%). Full labeled set 338/353 = 95.8%. Generated from `reports/tune/val-inspect.json` after the override (2026-09-13).

Each extra correct val call is ~1.4 points. Ceiling from these three, if you could save them all without breaking anything else, is 70/71 ≈ 98.6%. You almost certainly cannot.

Val class means, for scale: humans wait **1.22 s** median after the agent stops, synthetics **2.56 s**. Humans are messier (`rms_cv` 0.64 vs 0.25). Mixer weights acoustic **1.65** / behavioural **2.41** — timing wins ties.

---

## The three that remain

### 1. `call_6971b2685c1d` — human, both heads wrong

| | fused | acoustic | behavioural |
| --- | --- | --- | --- |
| p | 0.834 | 0.860 | 0.575 |
| kind | `miss_agree` | tidy spectrum | wait 1.75 s |

Listened. 97 s call, caller talks 19 s, agent 51 s. Four real waits: **1.50, 1.70, 1.79, 2.58 s**. Never snaps back (typical human min ~0.7 s). One barge-in vs human mean ~9. Quiet (speech RMS 0.03 vs a typical synth 0.21).

Close spectrogram of 9.4–14.7 s has real harmonic stacks — a careful person, not vocoder hash. Call-level stats still look like a bot: stable energy, almost no overlap.

**This is the only val call that needs a new cue.** Train has the same polite-human pattern: `4ac923`, `c6daaa`, `d80d9f`, `e0c1f6`.

### 2. `call_d4518fcb700a` — synthetic, mixer followed timing

| | fused | acoustic | behavioural |
| --- | --- | --- | --- |
| p | 0.332 | **0.870** (right) | 0.205 (wrong) |
| kind | `miss_mixer` | tidy, bot-like | wait 1.74 s (human-like) |

Acoustic already has it. Behavioural sees a human-length wait and high recovery CV (1.37). The mixer trusts timing more (weight 2.41 vs 1.65), so fused lands at 0.33.

`|0.332 − 0.5| = 0.168`, outside the 0.10 unsure band. Train analog: `cbad95` (fused 0.37, acoustic 0.84, behavioural 0.26).

### 3. `call_e2d538297f0c` — human, mixer followed acoustics

| | fused | acoustic | behavioural |
| --- | --- | --- | --- |
| p | 0.723 | 0.879 (wrong) | **0.459** (right, barely) |
| kind | `miss_mixer` | tidy | wait 1.51 s, recovery CV 0.89 |

Mixer already followed the confident head. Behavioural is 0.04 from the decision boundary. Trusting it means trusting 0.46 over 0.88 — the same max-margin idea that broke two other humans when applied to every disagreement.

### Already saved (do not re-litigate)

`call_569ffb0869eb` human. Mixer was 0.551, acoustic 0.021 (messy `rms_cv` 1.01), behavioural 0.947 (wait 2.50 s). Override took acoustic. That is the only val disagreement inside the unsure band.

---

## Key signals that actually work

1. **Positive-only floor-switch latency** (`caller_response_latency_pos_median_s`). Synthetics wait longer and more uniformly after the agent stops. Strongest behavioural separator. Live VAD, not organizer `turns.json`.
2. **Acoustic messiness, not loudness.** `rms_cv`, `zcr_std`, `spectral_flatness_std`, `centroid_std`, `crest_factor_cv`. Humans vary; TTS is stable. `caller_rms_mean` is louder-synth on this set and is an injection trap — dropped; stacked acc did not move.
3. **Two heads, then a mixer.** Behavioural 93% on live VAD; acoustic 87% without rms_mean; stacked 94.4% before override, 95.8% after. Concatenated looked better on some previews because it chewed collinear latency and rms_mean.
4. **Unsure-disagreement override only.** Naive max-margin on all 10 val disagreements: save 2 / break 2. Band 0.10: save 1 / break 0.

The brief’s useful hint is both channels: the agent’s turns tell you what the caller was reacting to. Floor-switch latency is that hint, measured.

---

## What not to do

**Do not add `caller_rms_mean`.** It would “save” 6971 because that caller is quiet. Synthetics in this dump are louder at a fixed digital level. Hidden set may not be.

**Do not ship concatenated.** 95.8–97% on some previews used organizer turns and/or rms_mean. Live path is VAD. Collinear `pos_mean` + `pos_median` inflates the number.

**Do not widen `disagree_unsure` toward 0.17 to catch d4518.** `bf038` sits at fused 0.356 (margin 0.144) and is currently correct. `9a3d` at 0.279 is a human the mixer got right while acoustic was 0.85 — max-margin would break it. Train `1761` (fused 0.611, acoustic 0.000, behavioural 1.000) is the 569ffb pattern just outside the band; still do not widen without simulating all 60 disagreements.

**Do not flip e2d538 by trusting the weaker head.** Behavioural 0.459 is not a signal; it is a coin flip that happened to land.

**Do not add barge-in, backchannel, or overlap to save 6971.** Those counts are 1 / 1 / 0.14 s vs human means ~9 / 8 / 5 s. They would call this person *more* synthetic.

**Do not add live Whisper / semantic.** Zero transcripts on disk. Three of five probes are the agent’s script. Unmeasured caller-only effect, seconds of latency. Depth beats breadth.

**Do not retrain on these three val rows.** Speaker-disjoint hidden set. Polite humans and fast bots will show up again; a model that memorizes 6971’s loudness will fail them.

**Do not dump the rest of the unused timing vector into the behavioural head.** `pos_mean` / `pos_min` / `pos_max` are the same wait, sliced differently. On 6971 they agree with `pos_median` or make it worse (fastest wait is already 1.50 s).

---

## Possible improvements (ranked)

These are the only honest next moves. None of them is required before judging.

1. **Freeze and serve.** Shipped path is the one that held on 353 labeled calls. Hidden-set risk is overfit, not underfit.
2. **Listen to d4518 and e2d538 the way we listened to 6971** before touching the mixer. If d4518’s caller *sounds* TTS (likely — acoustic 0.87) and merely answers quickly, there is no new feature; the mixer is doing what we asked (trust timing). If e2d538’s caller is actually messy on a cue we zeroed (`heavy=False` skips jitter, breath, formants), measure Cohen’s d on **val humans vs synths**, then on train, then decide. Do not add a heavy feature that only moves these two rows.
3. **Semantic, val-only, offline.** Transcribe val, score **caller** replies to trap questions / fillers. Keep it off `/detect` until the effect size is real and latency is acceptable. Not a 6971 fix: that call’s problem is wait + tidy spectrum, not (from the listen) invented account data.
4. **Intra-utterance pause structure** only if it separates polite humans from bots at the *call* level. 6971’s 5 s snippet has internal pauses the VAD glued into one turn. `caller_intra_silence_gap_*` already exists and on 6971 looks synthetic. Unlikely.

Do not treat train’s 12 misses as a punch list. Eight of them are mixer-wrong; four are agree-wrong polite humans. The override already encodes the only mixer case we could take without a wash.

---

## Revert

```
cp models/tandem-before-tune.json models/tandem.json   # pre-head-tune 90%
```

Override only: set `disagree_unsure` to `0` in `models/tandem.json`.

Listen files: `reports/tune/listen/6971-caller.wav`, `6971-agent.wav`.
Override table: `reports/tune/disagree-override.md`.
Full 353: `reports/tune/all-inspect.md`.
