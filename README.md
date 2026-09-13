# is-a-human

Human vs. synthetic caller detection for phone calls.

Live `POST /detect` demuxes stereo audio, runs hybrid VAD, scores acoustic and behavioural heads, and fuses them. Semantic is implemented but not in live fusion until transcripts exist at serve time.

## The problem

Given a recorded phone call between a caller and a bank's AI voice agent, decide whether the **caller** is a real person or a synthetic voice (ASR + language model + TTS dialing in).

Audio is stereo, 8 kHz, 16-bit PCM. **Channel 0 is the caller** — the one to classify. **Channel 1 is the agent.**

353 calls in Mexican Spanish, 282 train / 71 val, speaker-disjoint. Judging uses a hidden set of callers and voices in neither split.

## What we ship

A single HTTP endpoint:

```
POST /detect
{
  "call_id": "call_0181ce113ebe",
  "audio_base64": "<base64 of the complete WAV>",
  "sample_rate": 8000,
  "channels": 2
}
→ { "is_synthetic": true, "confidence": 0.87 }
```

`is_synthetic` is required. `confidence` is optional: certainty in the `is_synthetic` value (the judge recovers P(synthetic) as `confidence` when true and `1 - confidence` when false).

Stack, model, framework and hosting are all open.

## For reviewers

Read these first if you are reviewing the code:

| Path | Why |
| --- | --- |
| `src/is_a_human/api/app.py` | Judge contract: demux → score → `{is_synthetic, confidence}` |
| `src/is_a_human/api/schemas.py` | Request aliases (`audio_b64`) and how confidence is derived |
| `src/is_a_human/detect/live.py` | Live path: features then tandem predict |
| `src/is_a_human/detect/tandem.py` | Acoustic + behavioural heads, stacked fusion, disagreement + talkative-quiet gates |
| `src/is_a_human/analysis/features.py` | Feature dataclass and the shipped head feature lists |
| `src/is_a_human/turns/backends.py` | VAD backends; default is `hybrid` |
| `src/is_a_human/turns/energy_vad.py` | Fitted energy detector used at serve time |
| `models/tandem.json` | Trained artifact loaded at API startup (`is-a-human-train` writes it) |

**Live path.** Base64 WAV → `demux_base64_telephony` (ch0 caller, ch1 agent) → hybrid VAD ledger → acoustic + behavioural features from those segments → two logistic heads → stacked fusion on the head scores → optional overrides → `is_synthetic = P(synthetic) >= 0.5`. Live scoring uses `heavy=False`, so formant / pitch / interaction extras are skipped; they do not change the shipped verdict.

**VAD.** Organizer `turns/*.json` exist only for train/val. Serve time has audio only, so VAD must reproduce those segments. Default `hybrid` is fitted energy (14 dB over a 12th-percentile floor, 120 ms min speech, 280 ms merge gap) with Silero only if energy returns no segments.

**Heads.** Acoustic: `caller_rms_cv`, `caller_zcr_std`, `caller_crest_factor_cv`, `caller_spectral_flatness_std`, `caller_spectral_centroid_std`. Behavioural: `caller_response_latency_pos_median_s`, `agent_talk_ratio`, `agent_aligned_recovery_cv`. Mean loudness (`caller_rms_mean`) is treated as a likely injection artifact and is not in the shipped acoustic head.

**Fusion.** Stacked logistic on out-of-fold head scores. If the heads disagree and the mixer is within 0.10 of 0.5, the sharper head wins. A class-agnostic quiet-patient gate (wait 1.4–2.0 s and tidy `rms_cv`) plus a talkative subtype (≥ 24 caller turns) can flip a synthetic mixer call to human.

**Not in live fusion.** Semantic / transcript probes exist under `analysis/semantic.py` but are not scored unless transcripts are passed in. Do not assume they affect `/detect`.

**Constraints.** Never name the challenge sponsor. Never commit the dataset. Do not try to identify callers.

## Repo layout

| Path | Contents |
| --- | --- |
| `src/is_a_human/` | Application code |
| `src/is_a_human/audio/` | Base64/WAV demux and validation |
| `src/is_a_human/dataset/` | Local dataset loader (`manifest`, `audio`, `turns`) |
| `src/is_a_human/turns/` | Dual-channel VAD, turn ledger, metrics |
| `src/is_a_human/analysis/` | Feature extraction and offline exploration |
| `src/is_a_human/detect/` | Tandem train / predict and live scoring |
| `src/is_a_human/eval/` | Offline eval, VAD IoU, layer benchmarks |
| `src/is_a_human/api/` | FastAPI app and `/detect` |
| `models/tandem.json` | Trained tandem artifact (needed to serve) |
| `tests/` | Unit and integration tests |
| `demo/` | Local demo page served at `GET /` |
| `resources/hackmty26-main/` | Challenge manifest, turns, and judge check script |
| `resources/audio/` | Stereo WAV files (**gitignored**, add locally) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Train before serving if `models/tandem.json` is missing:

```bash
is-a-human-train
```

## Run locally

```bash
is-a-human-serve
```

`GET /health` should report `model_loaded: true`. Demo page: `http://localhost:8000/` (from `demo/index.html`, no build step). Drop stereo WAV calls to see the verdict, each head's vote, and round-trip time.

Score the live endpoint the same way the judge does (audio lives in `resources/audio/`):

```bash
.venv/bin/python resources/hackmty26-main/scripts/check_endpoint.py \
  --url http://127.0.0.1:8000/detect \
  --manifest resources/hackmty26-main/manifest.csv \
  --audio-dir resources/audio \
  --split val --n 20
```

`--n 0` runs the full val split. Same Wi‑Fi: replace the host with `$(ipconfig getifaddr en0)`. Off-network: `cloudflared tunnel --url http://localhost:8000 --protocol http2` (this network blocks QUIC; hostname changes every restart).

```bash
is-a-human-eval --split val          # VAD IoU vs organizer turns
is-a-human-explore --split val       # human vs synthetic feature comparison
is-a-human-benchmark                 # layer benches
is-a-human-train                     # writes models/tandem.json
pytest                               # all tests
pytest -m "not integration"          # fast unit tests only
```

## Dataset

Both folders are required locally and are not committed to git:

```
resources/
  hackmty26-main/    manifest.csv + turns/ (in repo)
  audio/             unzip the audio release here (call_<id>.wav)
```

Without the WAV files, eval and integration tests skip or fail.

Each `turns/<anon_id>.json` lists speech segments for both channels. Use them to validate VAD, not at serve time.

## Ground rules

- **Censor the sponsor's name.** Never named in code, comments, commit messages, docs, or file names. Write "the sponsor" or "the organizers". The repo is public.
- **Never commit the dataset.** Licensed for the hackathon only. `resources/audio/`, `dataset/`, `*.wav` and `*.zip` are gitignored — keep it that way.
- **Don't try to identify callers.** Human participants volunteered under recording notice and used invented personal data.
