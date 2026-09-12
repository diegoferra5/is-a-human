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
  - **Acoustic** — the audio itself. Originally deprioritized as engine-specific; it is in fact the strongest measured signal so far, but see the warning below before trusting it.
- The provided `turns/` JSON is ground truth for validating our own VAD — it does **not** exist at judging, so nothing that ships may read it.

## State of the code — read this before writing any

Three people built in parallel without seeing each other's work, so **two parallel codebases now exist side by side** on the `integration` branch. They have not been reconciled.

| | Max — `src/views/`, `src/features/` | Roger — `src/is_a_human/` |
| --- | --- | --- |
| VAD | own, energy-threshold | **Silero** (pretrained, torch) |
| Behavioral | latency, turn lengths, overlaps, talk ratio | recovery, interaction physics |
| Semantic | stub (wired to transcripts) | — |
| Acoustic | stub | **68 features, measured** |
| Fusion | **K-fold stacking over views** | single classifier |
| Tests | none | **15 files** |
| Result | none yet | **88.7% val acc, AUC 0.967** |

Roger's is further along and tested. Max's `View` interface (`fit` / `proba` / `load`) plus `fusion.py` is the better abstraction for three people working in parallel. Likely outcome: Roger's package as the base, Max's View/fusion on top. **Not yet decided — see `INTEGRATION.md`.**

Layer ownership as it actually stands: Max built VAD + behavioral. Roger built VAD + behavioral + acoustic. Nobody but Diego has touched semantic.

## Two things that will cost us if ignored

**`caller_rms_mean` is probably measuring the wrong thing.** Roger's top feature by a wide margin is how loud the caller is (d = +1.50 train, +2.26 val, rank 1/1 on both). That is far more likely an artifact of synthetic audio being injected at a fixed digital level than a property of synthetic speech — humans hold real handsets at varying distances. The hidden set uses unseen engines and possibly a different injection path, so it could weaken or flip. **Retrain without it and re-score before trusting 88.7%.** The variance features (`caller_rms_cv`, `caller_zcr_std`, `caller_crest_factor_cv`) measure messiness and are much safer.

**Speaker-disjointness is easy to lose.** `train` / `val` are speaker-disjoint by design. Any random K-fold over all 353 calls puts the same caller on both sides and inflates the score. Max's `fusion.py` does exactly this; check whether Roger's classifier does too.

## What the agent's script actually does

Found by reading transcripts. The script is near-identical across calls, so these are matchable deterministically.

1. **Digit trap.** The agent states a reference number, then later reads it back **altered by exactly one digit** and asks "¿es correcto?". Seen in 3 of the first 6 calls: `77120469 → 77120459`, `66412076 → 66412075`, `882311457 → 88231147`.
2. **False dilemma.** "¿Esto es sobre su cuenta Nómina Plus o sobre su crédito verde?" — a forced choice between two products where at least one presumably does not exist.
3. **Announced interruptions.** "Perdón que la interrumpa, [nombre]…" — the agent says out loud when it interrupts, so interruption points can be found by phrase match, not only timing overlap. **Useful to the behavioral layer.**

Caveat: those calls were all `synthetic`. The probes are solid (the agent script is identical for everyone), but the human-vs-machine *response* difference is still an untested hypothesis.

## Transcripts

`src/transcribe.py` (mlx-whisper `large-v3-turbo`, Spanish, both channels, 8k→16k, resumable) writes `transcripts/<call_id>.json`:

```json
{"call_id": "...", "turns": [{"channel": 0, "start": 10.5, "end": 20.2, "text": "..."}]}
```

Turn order and timestamps are kept deliberately: the probes are *pairs* — what the agent said, and how the caller answered right after. Flattening each side into one blob of text destroys them.

**`transcripts/` must never be committed.** The agent speaks the sponsor's bank name in the opening line of every single call, so every transcript contains it. Already gitignored — watch for notebook output cells and pasted excerpts.

Full run takes ~2 h on an M2. Never trigger it from inside `fit()`.

## Open / not yet decided

- Which codebase is the base, and which VAD ships
- Whether 88.7% survives dropping `caller_rms_mean`
- How the layers get combined into one score
- Deployment target for the endpoint
- Fourth team member's role

## Where things live

| Path | Contents |
| --- | --- |
| `INTEGRATION.md` | Temporary. The merge brief — what to keep from each codebase, and in what order. Delete when done. |
| `kb/` | Challenge brief and notes |
| `docs/` | `roadmap.html`, `deck.html` — published as artifacts |
| `reports/` | Roger's measured findings, feature rankings, charts |
| `src/views/`, `src/features/` | Max's scaffold |
| `src/is_a_human/` | Roger's package |
| `src/transcribe.py` | Whisper transcription |
| `dataset/`, `transcripts/` | gitignored, never commit |
