# is-a-human

Human vs. synthetic caller detection for phone calls — a hackathon challenge entry.

**Status:** exploration. Dataset cloned, challenge understood. No approach chosen yet, no code written yet.

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
| `kb/` | Knowledge base — challenge brief and working notes |
| `dataset/` | Challenge dataset (cloned separately, **gitignored**) |

## Dataset

Not in this repo. Clone it into `dataset/` and pull the audio zip from that repo's Releases:

```bash
git clone <challenge-dataset-repo> dataset
# then unzip the audio release so files land in dataset/audio/
```

353 calls in Mexican Spanish, 282 train / 71 val, speaker-disjoint. Judging uses a hidden set of callers and voices in neither split.

## Ground rules

- **Censor the sponsor's name.** The challenge sponsor is never named anywhere in this repo — not in code, comments, commit messages, docs, or file names. Write "the sponsor" or "the organizers". The repo is public.
- **Never commit the dataset.** It is licensed for the hackathon only and must not be redistributed. `dataset/`, `audio/`, `*.wav` and `*.zip` are gitignored — keep it that way.
- **No confidential source material.** The original challenge PDF stays local (`kb/source/`, gitignored). Only our own transcribed, scrubbed notes get committed.
- **Don't try to identify callers.** Human participants volunteered under recording notice and used invented personal data.

## Notes

See `kb/01-challenge-brief.md` for the full challenge breakdown — deliverable, dataset shape, suggested signals and judging criteria.
