# VAD

Voice activity detection for stereo 8 kHz telephony: **channel 0 = caller**, **channel 1 = agent**.

At judging we receive audio only. Organizer `turns/*.json` files exist for local train/val and are the scoring target for this VAD. They will not be present at serve time. Serve-time VAD must reproduce those `{channel, start, end}` segments so downstream features match train-time features.

## Default

`hybrid` (also the pipeline default in `backends.py`).

On this dataset hybrid is **fitted energy VAD**, with Silero timestamps only if energy returns no segments.

```bash
is-a-human-vad-benchmark --split val
```

Metric: interval IoU vs organizer JSON, per channel. Not human-vs-synthetic accuracy.

## Lineage

| Source | Detector | Notes |
| --- | --- | --- |
| Roger | Preloaded **Silero** (`torch.hub` / `silero-vad`) | Strong on the clean agent channel; conservative on the noisy caller |
| Max | Custom **energy** threshold (`src/vad.py`) | 25 ms frames, 10 ms hop, originally 6 dB above the 10th-percentile floor |
| This branch | Pluggable backends + shared postprocess | `silero`, `energy`, `hybrid` |

The first hybrid on this branch was **caller = Silero AND energy, agent = Silero only**. That fused the two detectors. It is **not** the current default — see below.

## Ground truth

JSON under `turns/<anon_id>.json`:

```json
{"turns": [{"channel": 0, "start": 5.32, "end": 5.64}, {"channel": 1, "start": 0.34, "end": 2.24}]}
```

Times are seconds, two decimal places (10 ms). Overlap across channels is allowed.

## What the labels look like

On a 20-call val subset, before the energy fit:

| backend | caller prec / rec | agent prec / rec | caller pred/ref duration |
| --- | --- | --- | --- |
| Silero @ 0.5 | 0.90 / 0.86 | 0.96 / 0.98 | ~1.00 |
| Energy @ 6 dB | 0.83 / 0.99 | 0.73 / 1.00 | ~1.43 |
| Old hybrid (AND @ 0.5) | 0.93 / 0.85 | 0.96 / 0.99 | ~0.95 |

Energy at 6 dB almost never missed speech, but marked a lot of non-speech. Silero was balanced on duration and missed caller speech. AND-gating Silero onto energy **raised precision and cut recall** — the opposite of what the JSON needed.

Any Silero overlay on *fitted* energy also hurt caller IoU (even `silero >= 0.05`). The organizer turns behave like an energy VAD, not a neural VAD.

## Fitted energy (current detector)

Tuned against JSON IoU on **train (282 calls)**, confirmed on **val (71)**. Shared postprocess: drop segments shorter than 120 ms, merge gaps shorter than 280 ms.

| Parameter | Old Max default | Fitted | Why |
| --- | --- | --- | --- |
| Frame / hop | 25 ms / 10 ms | unchanged | Matches JSON 10 ms grid |
| Threshold | 6 dB above floor | **14 dB** | 6 dB over-detects; 16 dB starts to miss |
| Noise floor | 10th percentile | **12th percentile** | Small, consistent gain on both splits |
| Min speech | 120 ms | 120 ms | Already matched the labels |
| Min gap | 200 ms | **280 ms** | Organizer merges slightly longer pauses |

Energy threshold on train (then the same pattern on val):

| threshold | train overall IoU | val overall IoU |
| --- | --- | --- |
| 6 dB | 0.80 | ~0.81 |
| 10 dB | 0.93 | 0.93 |
| **14 dB** | **~0.95** | **~0.95** |
| 16 dB | 0.94 | 0.94 |

13–15 dB and 250–300 ms gaps sit on a plateau. 14 dB / 280 ms / 12th percentile is in the middle of that plateau, not a one-call spike.

## Silero backend

Still available as `--backend silero`. It now uses Silero’s timestamp API instead of a raw 0.5 mask:

- `threshold=0.5`, `neg_threshold=0.35` (hysteresis)
- `min_speech_duration_ms=120`, `min_silence_duration_ms=280`
- `speech_pad_ms=0`, `time_resolution=2` (centiseconds, like the JSON)

That improved Silero slightly. It did not catch energy once energy was fitted to the labels.

## Val results (71 calls)

Interval IoU vs organizer JSON.

| backend | caller | agent | overall | mean latency |
| --- | --- | --- | --- | --- |
| Silero (legacy timestamps) | 0.811 | 0.941 | 0.876 | ~790 ms |
| Energy @ 6 dB | 0.865 | 0.748 | 0.807 | ~10 ms |
| Old hybrid (AND) | 0.830 | 0.952 | 0.891 | ~1000 ms |
| Silero (telephony timestamps) | 0.819 | 0.948 | 0.884 | ~770 ms |
| **Energy / hybrid (fitted)** | **0.972** | **0.987** | **0.979** | **~9 ms** |

On val, hybrid matched energy exactly — the Silero fallback did not fire.

## Is this overfitting?

Not in the usual sense (not a 15-call needle). Train and val agree on the direction and on the plateau.

It **is** matched to the organizer’s turn recipe. That is the right target if serve-time VAD should reproduce the segments used to train features. It is the wrong target if hidden audio is much quieter, or if the hidden labels were produced some other way.

The bet: judging cares that our turns look like theirs, because theirs are what behavioral/acoustic features were built on.

## Code

| Path | Role |
| --- | --- |
| `src/is_a_human/turns/energy_vad.py` | Fitted energy detector |
| `src/is_a_human/turns/silero_vad.py` | Preloaded Silero + timestamp params |
| `src/is_a_human/turns/postprocess.py` | Mask → segments (120 ms / 280 ms) |
| `src/is_a_human/turns/backends.py` | `silero` / `energy` / `hybrid` |
| `src/is_a_human/eval/vad_benchmark.py` | JSON IoU benchmark |
| `src/vad.py` | Max’s original energy VAD (6 dB / 200 ms) — **not** the pipeline default |

```bash
# All backends on val
is-a-human-vad-benchmark --split val

# One backend, limited sample
is-a-human-vad-benchmark --backend energy --split val --limit 10

# Explicit dataset root
is-a-human-vad-benchmark --dataset-root resources/hackmty26-main --split val
```
