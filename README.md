# is-a-human

Human vs. synthetic caller detection for phone calls — a hackathon challenge entry.

**Status:** Phase 0 foundations in place — audio ingest, turn pipeline, eval harness, and `POST /detect` stub.

## The problem

Given a recorded phone call between a caller and a bank's AI voice agent, decide whether the **caller** is a real person or a synthetic voice (a speech-recognition + language-model + text-to-speech stack dialing in).

Audio is stereo, 8 kHz, 16-bit PCM. **Channel 0 is the caller** — the one to classify. **Channel 1 is the agent.**

## What we have to ship

A single HTTP endpoint, live during judging:

```
POST /detect   ← stereo WAV, 8 kHz, base64-encoded
→ { "is_synthetic": true, "confidence": 0.87 }
```

`is_synthetic` is required. `confidence` is optional and rewards calibration.

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
| `resources/hackmty26-main/` | `manifest.csv` and `turns/` (**gitignored**, you must add locally) |
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
# API (POST /detect + GET /health)
is-a-human-serve

# Foundation eval on val split (requires dataset/)
is-a-human-eval --split val

# Tests
pytest
```

## Current approach (Phase 0)

The live endpoint is contract-compliant but returns a **placeholder verdict** (`is_synthetic=false`, `confidence=0.5`) while we build detection signals on top of the foundation pipeline:

1. In-memory base64 stereo WAV demux (`ch0` caller, `ch1` agent)
2. Dual Silero VAD and aligned turn ledger (speech / overlap / silence)
3. Conversation metrics (talk time, overlap, response latency)
4. VAD validation against organizer `turns/*.json` via offline eval

## Dataset

**Both folders are required locally** and are not committed to git. Place them inside `resources/`:

```
resources/
  hackmty26-main/     ← clone or copy the challenge metadata repo here
    manifest.csv      # anon_id, label, split, duration_s
    turns/            # per-call VAD segments (channel, start, end)
  audio/              ← unzip the audio release here (call_<id>.wav, stereo 8 kHz)
```

1. Put `hackmty26-main/` (manifest + turns) in `resources/hackmty26-main/`
2. Put the WAV files in `resources/audio/`

The loader auto-detects this layout. Without both folders, eval and integration tests will skip or fail.

353 calls in Mexican Spanish, 282 train / 71 val, speaker-disjoint. Judging uses a hidden set of callers and voices in neither split.

Each `turns/<anon_id>.json` file lists speech segments for both channels — useful as a reference when validating our VAD, and as a starting point for turn-aligned analysis.

## Ground rules

- **Censor the sponsor's name.** The challenge sponsor is never named anywhere in this repo — not in code, comments, commit messages, docs, or file names. Write "the sponsor" or "the organizers". The repo is public.
- **Never commit the dataset.** It is licensed for the hackathon only and must not be redistributed. `resources/hackmty26-main/`, `resources/audio/`, `dataset/`, `*.wav` and `*.zip` are gitignored — keep it that way.
- **No confidential source material.** The original challenge PDF stays local (`kb/source/`, gitignored). Only our own transcribed, scrubbed notes get committed.
- **Don't try to identify callers.** Human participants volunteered under recording notice and used invented personal data.

## Notes

See `kb/01-challenge-brief.md` for the full challenge breakdown — deliverable, dataset shape, suggested signals and judging criteria.
