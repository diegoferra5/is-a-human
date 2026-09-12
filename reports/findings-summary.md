# Human vs. Synthetic Caller — Findings Summary

Plain-language guide to what we measured, what separated the classes, and what to do next.

**Dataset:** 353 phone calls (282 train / 71 val), stereo 8 kHz, Mexican Spanish.  
**Analysis:** 68 numeric features per call, compared human vs. synthetic callers.  
**Classifier:** Logistic baseline on stable top features → **88.7% val accuracy**, AUC **0.967**.

---

## The big picture

Real human callers sound **messier**. Their volume, brightness, and timing jump around more from moment to moment. Synthetic callers sound **louder but smoother** — like a pipeline holding everything at a steady level.

The strongest pattern across train and val:

| Pattern | Human | Synthetic |
|---|---|---|
| Average loudness | Quieter | **Louder** |
| Loudness variation | **High** (messy) | Low (smooth) |
| Voice texture variation | **High** | Low |
| Pitch / timing variation | **Higher** | Lower |

**Stable separators** (rank high on both train and val): `caller_rms_mean`, `caller_zcr_std`.

**Recommended model features:** `caller_rms_mean`, `caller_zcr_std`, `caller_rms_cv` — plus optional `caller_crest_factor_cv` and spectral std features as secondary inputs.

---

## Recommendations at a glance

| Verdict | Signals |
|---|---|
| **Keep** | Core acoustic micro-variance (`caller_rms_*`, `caller_zcr_std`, `caller_crest_factor_cv`, spectral std features) |
| **Keep as secondary** | Conversational timing (`agent_talk_ratio`, `caller_response_latency_cv`), recovery CV (`agent_aligned_recovery_cv`) |
| **Improve before shipping** | HNR (`caller_hnr_std`), pitch std (`caller_f0_std`), formant features (val-strong but direction suspicious at 8 kHz) |
| **Drop for now** | Breath detection, room bleed / echo, entrainment, barge-in decay, pitch jitter, pause entropy, LFCC |

---

## Signal catalog

Each entry explains **what we measure**, **in simple terms**, and **how well it separated** human vs. synthetic (Cohen's d on train / val; rank out of 68). Negative d means humans score higher; positive d means synthetics score higher.

---

### Core acoustic — loudness and texture

These measure the caller's voice on channel 0 during speech segments.

| Signal | What it measures (simple) | Train d / Val d | Rank (train / val) | Verdict |
|---|---|---|---|---|
| `caller_rms_mean` | **How loud the caller is on average.** Synthetic voices tend to sit at a higher, steadier volume. | +1.50 / +2.26 | 1 / 1 | **Keep** |
| `caller_rms_std` | **How much loudness jumps around** across speech chunks (absolute spread). | — | — | Keep (related to CV) |
| `caller_rms_cv` | **Relative loudness messiness** — humans vary more call-to-call within a conversation. | −1.37 / −1.56 | 4 / 6 | **Keep** |
| `caller_zcr_mean` | **Average “buzziness”** of the waveform (how often it crosses zero). | — | — | Secondary |
| `caller_zcr_std` | **How much that buzziness varies** — humans are less uniform chunk-to-chunk. | −1.37 / −1.81 | 3 / 3 | **Keep** |
| `caller_spectral_centroid_mean` | **Average brightness** of the voice (higher = sharper / brighter). | — | — | Secondary |
| `caller_spectral_centroid_std` | **How much brightness varies** — humans shift tone more. | −1.36 / −1.47 | 5 / 8 | **Keep** |
| `caller_spectral_flatness_mean` | **How “noisy vs. tonal”** the voice sounds on average. | — | — | Secondary |
| `caller_spectral_flatness_std` | **Variation in noisiness** — humans are less consistent. | −1.43 / −1.44 | 2 / 9 | **Keep** |
| `caller_hf_lf_ratio_mean` | **Balance of high vs. low frequencies** in the caller voice. | — | — | Secondary |
| `caller_hf_lf_ratio_std` | **Variation in that balance** across speech chunks. | — | — | Secondary |
| `caller_crest_factor_mean` | **Peakiness** — how spiky the waveform is vs. its average energy. | — | — | Secondary |
| `caller_crest_factor_std` | **Variation in peakiness** across chunks. | — / −1.48 | — / 7 | Keep (val-strong) |
| `caller_crest_factor_cv` | **Relative spikiness messiness** — humans vary more. | −1.28 / −1.82 | 8 / 2 | **Keep** |
| `caller_intra_silence_gap_mean_s` | **Average length of pauses** inside caller speech. | — | — | Secondary |
| `caller_intra_silence_gap_cv` | **How irregular those internal pauses are.** | — | — | Secondary |
| `caller_segment_length_cv` | **How much caller utterance lengths vary.** | — | — | Secondary |

---

### Micro-variation — pitch, formants, HNR, pauses

Finer-grained “biological messiness” signals added in Phase 1b.

| Signal | What it measures (simple) | Train d / Val d | Rank (train / val) | Verdict |
|---|---|---|---|---|
| `caller_formant_f1_std` | **How much the first vowel resonance (F1) moves around** — throat/mouth shape changing. | +0.78 / +1.73 | 21 / 4 | Caution — synthetic ranks *higher* on val; may be LPC artifact at 8 kHz |
| `caller_formant_f2_std` | **Variation in second resonance (F2)** — tongue position shifts. | +0.78 / +1.73 | 22 / 5 | Caution (same as F1) |
| `caller_formant_f3_std` | **Variation in third resonance (F3).** | — | — | Caution |
| `caller_formant_volatility_mean` | **Average formant movement** across F1/F2/F3. | +0.58 / +1.36 | 33 / 11 | Improve / validate |
| `caller_pitch_jitter` | **Tiny frame-to-frame pitch wobble** — natural vocal cord micro-movement. | −0.53 / −0.01 | 37 / 65 | Drop |
| `caller_pitch_shimmer` | **Tiny frame-to-frame volume wobble** between pitch cycles. | +0.44 / +0.51 | 40 / 44 | Drop |
| `caller_f0_std` | **How much overall pitch varies** during speech. | −1.05 / −0.97 | 10 / 17 | Improve — promising but not stable top-5 |
| `caller_f0_cv` | **Relative pitch variability.** | −0.21 / +0.26 | 52 / 51 | Drop |
| `caller_hnr_mean` | **Average harmonic-to-noise ratio** — voice clarity vs. breathiness. | −0.19 / −0.92 | 55 / 19 | Improve |
| `caller_hnr_std` | **How much clarity vs. noise shifts** across phonemes — humans should vary more. | −0.40 / −0.64 | 44 / 34 | Improve |
| `caller_hnr_cv` | **Relative HNR variability.** | +0.05 / +1.12 | 64 / 15 | Improve — val-only spike |
| `caller_pause_entropy` | **Randomness of silence gap lengths** — human pauses less predictable. | −0.07 / −0.27 | 62 / 50 | Drop |
| `caller_lfcc_delta_delta_std` | **High-frequency spectral change rate** (vocoder frame artifacts). | −0.21 / −0.77 | 51 / 26 | Drop |

---

### Cross-channel interaction — bleed, breath, entrainment, interruptions

Uses both channel 0 (caller) and channel 1 (agent).

| Signal | What it measures (simple) | Train d / Val d | Rank (train / val) | Verdict |
|---|---|---|---|---|
| `caller_agent_bleed_correlation` | **Room echo:** agent voice leaking into caller mic during agent-only speech. Real phones pick up faint bleed; virtual audio paths may not. | −0.37 / +0.12 | 46 / 56 | Drop — no separation |
| `caller_agent_echo_correlation` | **Whole-call cross-correlation** between channels (unconditioned bleed). | −0.17 / −0.05 | 56 / 61 | Drop |
| `caller_breath_event_rate` | **Breath-like sounds per minute** of caller speech (inhale detection). | 0.00 / 0.00 | 67 / 67 | Drop — no signal in data |
| `caller_breath_gap_ratio` | **Share of pre-speech gaps that look like inhalations** (pink-noise bursts before long utterances). | 0.00 / 0.00 | 68 / 68 | Drop — no signal in data |
| `cross_channel_energy_correlation` | **Entrainment:** do caller and agent loudness rise/fall together? | −0.11 / −0.18 | 60 / 52 | Drop |
| `caller_yield_decay_mean_db` | **When agent interrupts, how fast caller fades out** (first 150 ms). Humans trail off; bots may cut sharply. | −0.19 / −0.15 | 54 / 53 | Drop |
| `caller_yield_decay_std_db` | **Variation in that fade-out behavior** across interruptions. | −0.14 / −0.28 | 58 / 49 | Drop |

---

### Conversational timing — who talks when

Derived from turn-taking on the call timeline.

| Signal | What it measures (simple) | Train d / Val d | Rank (train / val) | Verdict |
|---|---|---|---|---|
| `caller_talk_ratio` | **Fraction of the call the caller is speaking.** Synthetics often talk more. | +0.80 / +1.19 | 20 / 13 | Secondary |
| `agent_talk_ratio` | **Fraction of the call the agent is speaking.** Humans leave more room for the agent. | −1.26 / −1.44 | 9 / 10 | Secondary |
| `overlap_ratio` | **How much both sides talk at once.** | +0.41 / +0.47 | 42 / 46 | Drop |
| `silence_ratio` | **How much dead air** is in the call. | +0.66 / +0.62 | 28 / 36 | Secondary |
| `caller_segment_count` | **How many separate caller speech chunks.** | — | — | Secondary |
| `agent_segment_count` | **How many agent speech chunks.** | — | — | Secondary |
| `overlap_event_count` | **Number of overlap events.** | — | — | Secondary |
| `caller_utterance_mean_s` | **Average caller utterance length.** | — | — | Secondary |
| `caller_utterance_std_s` | **Variation in caller utterance length.** | — | — | Secondary |
| `agent_utterance_mean_s` | **Average agent utterance length.** | — | — | Secondary |
| `agent_utterance_std_s` | **Variation in agent utterance length.** | — | — | Secondary |
| `caller_response_latency_mean_s` | **Average delay before caller responds** after agent stops. | — | — | Secondary |
| `caller_response_latency_std_s` | **Spread in caller response delays.** | — | — | Secondary |
| `caller_response_latency_cv` | **Relative inconsistency in caller response timing** — humans vary more. | −1.30 / −0.96 | 7 / 18 | Secondary |
| `agent_response_latency_mean_s` | **Average agent response delay.** | — | — | Secondary |
| `agent_response_latency_std_s` | **Spread in agent response delays.** | — | — | Secondary |
| `agent_response_latency_cv` | **Relative inconsistency in agent timing.** | — | — | Secondary |
| `duration_s` | **Total call length in seconds.** | — | — | Not for classification |

---

### Recovery after turbulence — how caller reacts to agent

Measures caller behavior after the agent stops, overlaps, or interrupts (uses organizer turn boundaries).

| Signal | What it measures (simple) | Train d / Val d | Rank (train / val) | Verdict |
|---|---|---|---|---|
| `agent_aligned_recovery_cv` | **How consistently the caller responds** after the agent finishes a turn — humans vary more. | −1.32 / −1.31 | 6 / 12 | Secondary — good train, weaker val |
| `agent_aligned_recovery_mean_s` | **Average delay before caller speaks** after agent stops. | +0.89 / +0.63 | 14 / 35 | Secondary |
| `agent_aligned_recovery_std_s` | **Spread in those recovery delays.** | — | — | Secondary |
| `agent_aligned_recovery_count` | **How many agent-stop → caller-start events** we measured. | — | — | Secondary |
| `agent_aligned_recovery_utterance_mean_s` | **Average length of caller's first utterance** after agent stops. | — | — | Secondary |
| `agent_aligned_recovery_utterance_std_s` | **Variation in that utterance length.** | — | — | Secondary |
| `agent_aligned_recovery_utterance_cv` | **Relative variation in recovery utterance length.** | — | — | Secondary |
| `post_overlap_recovery_mean_s` | **Delay after overlap ends** before caller speaks again. | — | — | Secondary |
| `post_overlap_recovery_cv` | **Consistency of post-overlap recovery timing.** | −0.86 / −0.78 | 18 / 25 | Secondary |
| `post_overlap_recovery_count` | **Number of post-overlap recovery events.** | — | — | Secondary |
| `caller_barge_in_count` | **Times caller talked over the agent.** | −0.89 / −0.75 | 12 / 27 | Secondary |
| `caller_backchannel_count` | **Short caller utterances during agent speech** (backchannels). | — | — | Secondary |
| `agent_turn_fragmentation` | **How choppy the agent's turns are** (segments per unit time). | — | — | Secondary |

---

## What we learned from experiments that did not work

**Breath and room bleed:** Hypothesized that humans inhale audibly and pick up agent echo in the room. On this 8 kHz telephony dataset, both features scored **d = 0.00** — likely band-limited audio and short banking utterances hide breaths; call conditions vary too much for bleed.

**Formant volatility:** Ranks high on val but points the **wrong way** (synthetic > human). At 8 kHz, LPC formant tracking is unreliable — treat as research-only until validated on VAD segments.

**Expanded feature set vs. simpler model:** Adding 20 new features did not improve stable cross-split agreement (rank correlation fell from 0.818 → 0.727). More features ≠ better generalization here.

---

## Suggested next steps

1. **Ship a 3-feature model:** `caller_rms_mean`, `caller_zcr_std`, `caller_rms_cv` — force these as stable features rather than auto-selecting only 2.
2. **Switch acoustic extraction to VAD segments** for live `/detect` (currently uses organizer turns in batch mode).
3. **Persist model + write `/detect`** with real verdicts.
4. **Drop** breath, bleed, entrainment, and barge-in decay from the production pipeline to save ~30% extraction time.

---

## Source reports

- Full rankings: `reports/feature-rankings.txt`
- Bootstrap + classifier: `reports/explore-multi-v3.md`
- Original baseline (48 features): `reports/explore-multi.md`

*Generated 2026-09-12 from 353-call exploratory analysis.*
