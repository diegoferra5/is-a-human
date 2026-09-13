# Step 0 — is the agent's script the same, and is it fair?

Run: `python src/semantic/step0_script_check.py` · train split, 282 calls · raw table in `analysis/step0_beats.csv`

## 0a — the script skeleton is stable

Every call opens with the identical greeting, then name -> reference number. After that
the middle beats (announced interruption, false dilemma, confirmation) appear in varying
order, and several are optional. 77 distinct orderings across 282 calls, but that is mostly
beats *missing*, not beats *reordered*: `ask_ref` precedes `confirm_ref` in 94% of calls
where both fire.

**Conclusion:** match on phrases, never on position. Do not assume a beat exists.

## 0b — the script is NOT fair between the two groups

| beat | human | synthetic | gap |
|---|---|---|---|
| greeting | 100.0% | 100.0% | 0.0 |
| ask_ref | 100.0% | 99.4% | +0.6 |
| interrupt | 78.8% | 75.7% | +3.0 |
| ask_name | 85.0% | 78.1% | +6.8 |
| is_correct | 21.2% | 14.8% | +6.4 |
| folio | 38.9% | 33.1% | +5.8 |
| **dilemma** | **85.8%** | **71.0%** | **+14.8** |
| **confirm_ref** | **87.6%** | **68.6%** | **+19.0** |

The two most useful traps fire markedly more often on human callers. The gap is not
explained by call duration — it survives inside every length quartile (dilemma +5.7 to
+23.1, confirm_ref +0.8 to +28.7).

## Why: synthetic callers crowd the agent out

Same call length, same number of caller turns — but the synthetic caller says far more.

| | caller turns | caller words | words/turn | agent turns | agent words |
|---|---|---|---|---|---|
| human | 14.1 | 69.6 | 5.1 | 26.7 | 265.5 |
| synthetic | 14.5 | 107.2 | 7.4 | 22.1 | 213.5 |

54% more words in the same 88 s of talk time. The agent gets ~4.5 fewer turns and the
script advances less far, so the late traps never fire.

## Consequences for the layer

1. **"The trap fired" must never be a feature.** It leaks the label without measuring the
   caller at all, and it leaks for a reason that may not survive a different engine.
2. **Condition on the trap firing.** Features are only computed over calls where the beat
   is present, and every feature carries its own denominator.
3. **Verbosity is a real candidate** (5.1 vs 7.4 words per turn) — but see the warning.

## Warning: the word counts may be an ASR artifact

Whisper transcribes clean synthetic speech more completely than noisy human phone audio.
Some of the 54% word gap could be transcription quality, not talkativeness. Before using
any word-count feature, check it against turn *durations* from the audio, which do not
depend on the transcript. Caller talk time is nearly identical (87.6 s vs 88.4 s), which
is itself a reason for suspicion.
