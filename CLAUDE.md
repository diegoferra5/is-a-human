# is-a-human

Detect whether the caller on a recorded bank phone call is a real human or a synthetic voice.
Hackathon entry. Team: **Backward Deployed Engineers** — 4 people, 36 hours.

## Ground rules for this repo

- **Never name the sponsor or the hackathon** in anything committed — code, comments, commit messages, docs, filenames. Write "the sponsor" / "the organizers" / "the hackathon". The GitHub repo is public.
- **Never commit the dataset.** Licensed for the event only, no redistribution. `dataset/`, `audio/`, `*.wav`, `*.zip` are gitignored.
- The original challenge PDF stays local in `kb/source/` (gitignored). Only our own scrubbed notes are committed.
- Don't try to identify callers. Participants volunteered and used invented personal data.

## What we have to ship

One HTTP endpoint, reachable during judging:

```
POST /detect   ← stereo WAV, 8 kHz, base64-encoded
               channel 0 = caller (classify this one), channel 1 = agent
→ { "is_synthetic": true, "confidence": 0.87 }
```

`is_synthetic` required. `confidence` optional, used for tie-breaks and calibration.
Stack, model, framework, hosting all free. Plus a short README on the approach.

## Dataset

Cloned locally into `dataset/` (gitignored). Audio ships as a separate release zip → unzip so files land in `dataset/audio/`.

| | train | val | total |
|---|---|---|---|
| human | 113 | 37 | 150 |
| synthetic | 169 | 34 | 203 |
| | 282 | 71 | **353** |

- Stereo, 8 kHz, 16-bit PCM. Duration 61–273 s, mean ~148 s.
- `manifest.csv` — `anon_id`, `label`, `split`, `duration_s`
- `turns/<anon_id>.json` — `{"turns":[{"channel":0,"start":10.5,"end":20.16}]}`. **Only those three fields, no metrics.** Auto-derived. ~35 turns per call.
- Channels are already separated in the stereo file. No diarization needed.
- `train` / `val` are speaker-disjoint. Judging uses a hidden set of callers and voices in neither split.
- Synthetic callers are full stacks (speech recognition + language model + synthetic voice) dialing in, not audio playback.
- Every call follows the same customer-service flow. The agent interrupts, falls silent, asks callers to repeat information back, and sometimes asks about things that do not exist.

## Decisions made

- **Offline first.** Build the detector on recorded audio. A real-time demo with a live agent comes after, and only if the offline model is solid.
- **Three layers**, per the challenge's own framing:
  - **Conversational behavior** — turn timing, how the caller handles interruptions and silence
  - **Semantic** — what the caller says, especially when asked about things that do not exist
  - **Acoustic** — the audio itself. Deprioritized: engine-specific, and the hidden set uses unseen engines.
- **Roles:** one person on turn detection (VAD per channel), one on the behavioral layer, Diego on semantic.
- Behavioral and semantic coordinate on how features get extracted from the audio.
- The provided `turns/` JSON is usable as ground truth to validate our own VAD against.

## Open / not yet decided

- Fourth team member's role
- How the layers get combined into one score
- Deployment target for the endpoint

## Where things live

| Path | Contents |
| --- | --- |
| `kb/` | Challenge brief and notes |
| `docs/` | `roadmap.html`, `deck.html` — published as artifacts |
| `dataset/` | Challenge dataset, gitignored |
