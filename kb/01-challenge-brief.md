# Challenge: Defend the Bank Against Voice Deepfakes

Working notes on the challenge track. Original brief PDF and dataset are kept local (gitignored).

## The ask, in one line
Given a recorded phone call between a caller and a bank's AI voice agent, decide whether **the caller** is a real human or a synthetic (cloned/TTS) voice.

## Context
The sponsor runs AI voice agents for LatAm banks (MX, CO, PE, CL, BR) — collections, customer service, outbound campaigns. Real Spanish conversations over regular phone lines.

Contact centers are attacked from both directions: fraudsters impersonate customers (account takeover) and impersonate banks (social engineering). Most contact centers today have no reliable way to tell whether the voice on the line belongs to a real person. The people most exposed are those who bank by phone because they can't or don't use apps.

## Deliverable (hard requirement)
One HTTP endpoint, reachable throughout the judging window:

```
POST /detect
body: stereo WAV clip, 8 kHz, base64-encoded
      channel 0 = caller (the one to classify)
      channel 1 = agent
->
{ "is_synthetic": true, "confidence": 0.87 }
```

- `is_synthetic` — **required** (boolean)
- `confidence` — optional but recommended; used for tie-breaks and to reward calibration
- Everything else free: model, language, framework, hosting
- Ship a short README explaining the approach

## Dataset
353 recorded phone conversations, Mexican Spanish.

| | train | val | total |
|---|---|---|---|
| human | 113 | 37 | **150** |
| synthetic | 169 | 34 | **203** |
| | **282** | **71** | **353** |

Duration 61–273 s, mean ~148 s, ~14.5 h total.

- Stereo, telephony-grade **8 kHz**, 16-bit PCM
- **Channel 0 = incoming caller** (classification target), **channel 1 = agent**
- Diverse by design: speakers, devices, accents, call conditions
- `manifest.csv` — `anon_id`, `label`, `split`, `duration_s`
- `turns/<anon_id>.json` — auto-derived VAD segments: `{"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}`
- `audio/<anon_id>.wav` — distributed separately as a release zip
- `train` and `val` are **speaker-disjoint**. Judging uses a hidden set of callers and voices absent from both splits.

Synthetic callers are full stacks — speech recognition + language model + synthetic voice — dialing the same number, not just cloned audio playback.

## Suggested attack surfaces (directions, not requirements)
1. **Acoustic** — the voice itself: spectral artifacts, prosody, breathing, vocoder/model fingerprints. Train or fine-tune a classifier.
2. **Conversational behavior** — how the caller handles conversational turbulence. Every call contains moments where the agent *interrupts, falls silent, or talks over* the caller. Humans recover instantly and messily; machines recover consistently. **Consistency is the signal.**
3. **Semantic** — what the caller actually says. The agent asks callers to repeat information back, and sometimes asks about **things that do not exist**. A human says "I don't have that." A language model tends to invent an answer.

Every call follows the same customer-service flow regardless of who is calling — so calls can be aligned to each other and caller responses compared at the same scripted prompt.

## Two explicit hints from the organizers
- You get **both channels for a reason**. The caller's audio is not the only signal; the agent's turns tell you *what the caller was reacting to*.
- **Depth beats breadth.** One signal done well outscores three that half-work.

## Evaluation
Judges visit each team for **15 minutes**. They run their benchmark against the live endpoint using a **hidden test set** while the team walks through the solution. The endpoint must stay reachable.

| Criterion | What it means |
|---|---|
| **Robustness** | Performance on unseen speakers, TTS engines, and call conditions |
| **Originality** | Signals beyond an off-the-shelf audio classifier |
| **Technical depth** | Quality of execution + does the team understand *why* it works |
| **Feasibility** | Could a bank actually deploy this on real phone audio |
| **Latency** | How fast a confident verdict is reached |

## Why it matters
Phone is still how a large part of LatAm banks, and for many customers it is the only channel they have. As synthetic voices become cheap and convincing, that channel either becomes verifiable or becomes untrustworthy — and the people who lose most are the ones with no alternative.

## Terms
Human callers volunteered, were told the call was recorded for an AI test, and used invented personal data. Do not try to identify anyone. The dataset is provided for the hackathon only and must not be redistributed.
