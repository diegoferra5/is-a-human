# Semantic layer — robustness pass

Branch `semantic-robust`. Nothing committed. All design decisions were made on
**train 5-fold OOF (3 seeds, 282 calls)**; val (71 unseen voices) was scored once
per configuration at the end and is reported, not tuned on.

## What changed and why

1. **Serve-consistent transcripts.** The semantic head was trained on whole-channel
   Whisper transcripts but served on VAD-trimmed ones. Retrained on transcripts
   produced by the exact serving path (`is_a_human.asr_batch`). Semantic head alone:
   train OOF 85.6 → **89.2**, val 83.1 → **87.3** (AUC 0.897 → 0.952).
2. **Disagreement trigger.** The gate opened only when the fast fusion was unsure —
   but a confidently *wrong* head never looks unsure, so the semantic head was never
   consulted exactly when needed. Now it is also consulted when the acoustic and
   behavioural heads contradict each other (each > 0.15 from 0.5 on opposite sides).
3. **Parameter-free combiner.** Inside the gate the answer is the mean of the log-odds
   of the fast fusion and the semantic head. A fitted three-input combiner scored +0.5
   on clean train OOF and 2–3 points *worse* whenever a head was masked or corrupted —
   it had learned the clean regime. Rejected.
4. **Head availability + masking.** When the VAD finds no caller speech (or < 1 s), the
   acoustic/behavioural heads are replaced by their fusion-neutral value, the gate is
   forced open, and Whisper gets the whole channel. Silent audio no longer crashes.
5. **Tried and rejected:** hard-example weighting of the semantic head (3× on gated
   rows): ±0.3, no effect. Lower disagreement margins (0.05/0.10): +1.6 on the flipped
   scenario for +3 points of latency; kept 0.15.

## Before vs after — train OOF (mean of 3 seeds)

| scenario | fast path only | before (unsure gate, fitted fusion₃) | **after** |
|---|---|---|---|
| clean | 95.7 | 97.4 | **97.5** |
| acoustic masked | 91.7 | 96.0 | 95.7 |
| behavioural masked | 86.9 | 90.0 | 91.0 |
| both masked | 59.9 | 84.2 | 87.8 |
| acoustic noisy | 94.0 | 96.0 | 96.3 |
| behavioural noisy | 91.8 | 94.3 | 95.2 |
| acoustic flipped | 76.0 | 85.9 | 86.2 |
| **behavioural flipped** | 26.1 | 44.2 | **68.0** |
| semantic masked | 95.7 | 96.0 | 95.7 |

"before" = the merged design with full-channel transcripts. Gated fraction 13.1% → 16.2%.

## Unseen-call performance (val, 71) — judge's own script, live endpoint

| | Roger fast path | before | **after** |
|---|---|---|---|
| accuracy | 95.8 | 94.4 | **94.4** |
| balanced accuracy | 95.8 | 94.6 | 94.6 |
| TPR synthetic / TNR human | 97.1 / 94.6 | 100 / 89.2 | **100** / 89.2 |
| AUC (from confidence) | 0.997 | 0.998 | **0.999** |
| Brier | 0.045 | 0.033 | 0.035 |
| mean / max latency | 69 ms / 0.16 s | 0.45 s / 11.7 s | **0.62 s / 7.8 s** |
| calls sent to Whisper | 0 | 5 | 10 |

Per call: the gate fixes one synthetic (d4518, fast 0.33 → 0.82) and breaks two
fast-talking humans (95cce, 9a3d5: semantic 0.77 / 0.91). Net −1 call vs the fast path
on val; +1.8 points on train OOF. One call on 71 is inside seed noise. The four remaining
misses are all humans, three of them answered at 0.64–0.66 — calibrated doubt, not
confident errors, which is what the Brier score rewards.

## Robustness when the fast heads fail (val, scored once)

| scenario | fast path | after |
|---|---|---|
| acoustic masked | 93.0 | 93.0 |
| behavioural masked | 85.9 | 88.7 |
| both masked | 47.9 | 84.5 |
| behavioural noisy | 88.7 | 95.8 |
| acoustic flipped | 85.9 | 88.7 |
| behavioural flipped | 22.5 | 67.6 |
| semantic masked | 94.4 | 94.4 |

With both fast heads dead the model still answers at 84.5% (semantic alone: 87.3%).
With the behavioural head confidently wrong it holds 67.6% instead of collapsing to
22.5%. Removing the semantic head costs nothing on clean data.

## Remaining weaknesses

- **Fast-talking humans.** The semantic head's backbone is words per second of speech;
  humans at 2.5+ words/s look synthetic to it (val: 95cce, 9a3d5). A second
  density-independent signal would be needed to fix this without overfitting.
- **Behavioural head flipped is still the worst case (68%).** The disagreement trigger
  only fires when the acoustic head disagrees *confidently*; when acoustic is also
  unsure, a wrong behavioural head still decides. Lowering the margin buys ~2 points
  per 3 points of latency.
- **Both fast heads wrong together** is not covered — the semantic head is outvoted.
- **Latency tail** is 7.8 s on gated calls (judge timeout 30 s); mean 0.62 s.
- Val has now been scored by four configurations across sessions. Treat further val
  numbers as reporting only.

Reproduce: `python -m is_a_human.eval.robustness --transcripts transcripts_trimmed --seeds 3 --val`
