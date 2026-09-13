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

The synthetic caller produces far more words per second of speech (1.51 vs 2.48 by the
shipped VAD, AUC 0.833) and, by the dataset's own turn files, takes *fewer* turns (21.9
vs 15.0). Longer, denser turns leave the agent fewer openings, so the script advances
less far and the late beats never fire.

*An earlier version of this section claimed "same turn count (14), same talk time (88 s)"
from Whisper segment spans and warned that the word gap might be an ASR artefact. Both
were wrong: Whisper spans include surrounding silence (2.2× true duration) and merge
turns, and the agent channel — the same TTS voice on every call — shows identical
words-per-second across groups, which rules out the ASR hearing the two groups
differently. See `analysis/facts.md` F3.*

## Consequences for the layer

1. **"The trap fired" must never be a feature.** It leaks the label without measuring the
   caller, for a reason that may not survive a different engine.
2. **Every feature is a property of the caller's own turns** — a rate, a per-call flag on
   what the caller said — never a presence flag on the agent's script.
3. **Speech density is real and is the layer's main signal.** It ships as `speech_rate`.
