# is-a-human

Human vs. synthetic caller detection for phone calls — a hackathon challenge entry.

**Status:** Live `POST /detect` runs hybrid VAD, scores acoustic + behavioural heads, and fuses them. Semantic is wired but not in live fusion until transcripts exist.

## The problem

Given a recorded phone call between a caller and a bank's AI voice agent, decide whether the **caller** is a real person or a synthetic voice (a speech-recognition + language-model + text-to-speech stack dialing in).

Audio is stereo, 8 kHz, 16-bit PCM. **Channel 0 is the caller** — the one to classify. **Channel 1 is the agent.**

## What we have to ship

A single HTTP endpoint, live during judging:

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

## Repo layout

| Path | Contents |
| --- | --- |
| `src/is_a_human/` | Application code |
| `src/is_a_human/audio/` | Base64/WAV demux and validation |
| `src/is_a_human/dataset/` | Local dataset loader (`manifest`, `audio`, `turns`) |
| `src/is_a_human/turns/` | Dual-channel VAD, turn ledger, metrics |
| `src/is_a_human/eval/` | Offline foundation eval harness |
| `src/is_a_human/api/` | FastAPI app and `/detect` endpoint |
| `tests/` | Unit tests |
| `kb/` | Knowledge base — challenge brief and working notes |
| `resources/challenge-dataset/` | `manifest.csv` and `turns/` (**gitignored**, you must add locally) |
| `resources/audio/` | Stereo WAV files (**gitignored**, you must add locally) |
| `dataset/` | Alternate layout if cloned as a single folder (**gitignored**) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run locally

```bash
# API (POST /detect + GET /health) — see SERVE.md for LAN / tunnel / judge URLs
is-a-human-serve

# Foundation eval on val split (requires dataset/)
is-a-human-eval --split val

# Exploratory analysis: human vs synthetic feature comparison
is-a-human-explore --split val

# Layer benchmarks → reports/benchmarks/index.html
is-a-human-benchmark
is-a-human-benchmark --limit 20          # faster subset
is-a-human-benchmark --suites vad        # VAD IoU only

# Train tandem detector (VAD → acoustic + behavioural → fusion)
is-a-human-train                         # writes models/tandem.json

# Tests
pytest                          # all tests + HTML report at reports/test-results/index.html
pytest -m "not integration"     # fast unit tests only
pytest -m vad                   # VAD suite
pytest -m acoustic              # acoustic suite
pytest -m semantic              # transcript probe suite
pytest -m behavioral            # turn-timing / recovery suite
```

## Current approach

`POST /detect` demuxes stereo audio, runs hybrid energy VAD, extracts acoustic + behavioural features from the VAD ledger, and fuses them. `is_synthetic` is `P(synthetic) >= 0.5`; `confidence` is certainty in that label. Train with `is-a-human-train` so `models/tandem.json` is present at serve time.

Score the live endpoint the same way the judge does (audio lives in `resources/audio/`):

```bash
is-a-human-serve
.venv/bin/python resources/hackmty26-main/scripts/check_endpoint.py \
  --url http://localhost:8000/detect \
  --manifest resources/hackmty26-main/manifest.csv \
  --audio-dir resources/audio \
  --split val --n 20
```

## Dataset

**Both folders are required locally** and are not committed to git. Place them inside `resources/`:

```
resources/
  challenge-dataset/     ← clone or copy the challenge metadata repo here
    manifest.csv      # anon_id, label, split, duration_s
    turns/            # per-call VAD segments (channel, start, end)
  audio/              ← unzip the audio release here (call_<id>.wav, stereo 8 kHz)
```

1. Put `challenge-dataset/` (manifest + turns) in `resources/challenge-dataset/`
2. Put the WAV files in `resources/audio/`

The loader auto-detects this layout. Without both folders, eval and integration tests will skip or fail.

353 calls in Mexican Spanish, 282 train / 71 val, speaker-disjoint. Judging uses a hidden set of callers and voices in neither split.

Each `turns/<anon_id>.json` file lists speech segments for both channels — useful as a reference when validating our VAD, and as a starting point for turn-aligned analysis.

## Ground rules

- **Censor the sponsor's name.** The challenge sponsor is never named anywhere in this repo — not in code, comments, commit messages, docs, or file names. Write "the sponsor" or "the organizers". The repo is public.
- **Never commit the dataset.** It is licensed for the hackathon only and must not be redistributed. `resources/challenge-dataset/`, `resources/audio/`, `dataset/`, `*.wav` and `*.zip` are gitignored — keep it that way.
- **No confidential source material.** The original challenge PDF stays local (`kb/source/`, gitignored). Only our own transcribed, scrubbed notes get committed.
- **Don't try to identify callers.** Human participants volunteered under recording notice and used invented personal data.

## Notes

See `kb/01-challenge-brief.md` for the full challenge breakdown — deliverable, dataset shape, suggested signals and judging criteria.
