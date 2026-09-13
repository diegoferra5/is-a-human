# Semantic layer — what we know, and how we know it

Numbers are **train split, 282 calls** (113 human, 169 synthetic) unless marked val.
"The agent" = the bank's scripted AI, channel 1. "The caller" = the one we classify,
channel 0. Model results and the val score live in `src/semantic/RESULTS.md`; this file is
the evidence behind each claim.

---

## CONFIRMED

### F1. The agent runs the same script on every call
Identical opening line word-for-word in all 353 calls, same agent name, same question set.
Beat order varies in the middle and beats are often skipped, but `ask_ref` precedes
`confirm_ref` in 94% of calls where both fire.
**How:** `src/semantic/step0_script_check.py`.
**So:** deterministic phrase matching generalises. Never match on position.

### F2. The script's later beats fire more often on human callers
| beat | human | synthetic |
|---|---|---|
| forced choice | 85.8% | 71.0% |
| confirm reference | 87.6% | 68.6% |
| altered read-back | 81.4% | 52.1% |

Survives inside every duration quartile, so it is not call length.
**So:** *"did the trap fire"* can never be a feature — it encodes the label without
measuring the caller. Every shipped feature is a rate or a per-call property of the
caller's own turns, not a presence flag on the agent's script.

### F3. The synthetic caller talks denser — and it is real, not the ASR
| | human | synthetic | AUC |
|---|---|---|---|
| caller words per second of VAD-measured speech | 1.51 | 2.48 | **0.833** |
| caller VAD turn count (dataset turn files) | 21.9 | 15.0 | 0.767 |
| caller VAD talk time (s) | 43.1 | 38.8 | 0.576 |
| **agent** words per second of VAD speech | 3.29 | 3.25 | 0.534 |

The last row is the control. The agent is the same TTS voice on every call, recorded
through the same path. If Whisper under-transcribed the human-call recordings, the agent's
rate would drop on those calls too. It does not move. **The word gap is caller behaviour.**
**How:** review of this layer, using `dataset/turns/` (audio-derived) as the denominator;
the shipped feature uses the behavioral view's VAD (0.833; 0.885 with the dataset's turns).
**So:** `speech_rate` is the layer's strongest feature and its backbone.

*Earlier versions of this file measured "talk time" and "turn count" from Whisper segment
spans and concluded they were equal across groups (88 s, 14 turns). Whisper spans include
the silence around speech — 2.2× the true duration — and its segmentation merges turns.
Both numbers were wrong; both are now taken from the VAD.*

### F4. With speech rate controlled, the behaviour features are an increment of about +0.08 AUC
| feature | AUC alone | residual AUC after `speech_rate` |
|---|---|---|
| `positional_correction` | 0.669 | ~0.51 |
| `closing_ritual` | 0.685 | ~0.53 |
| `calls_agent_by_name` | 0.672 | ~0.52 |
| `question_turn_rate` | 0.635 | ~0.72 |

Jointly they lift the model from 0.833 (`speech_rate` alone) to 0.914 (all 12). No single
one adds more than +0.02 on its own.
**How:** review; each feature regressed on `speech_rate`, AUC of the residual.
**So:** they are worth shipping — and they are the part a judge can *see* in a transcript —
but density is the mechanism, precision and politeness are the increment.

### F5. The altered read-back separates the groups by *how* the caller corrects, not whether
The agent misreads the reference number. Both groups notice at about the same rate. The
synthetic caller names the position ("termina en 06, no 05", "el último es 5"); the human
says "está mal" and re-dictates the whole number.
`positional_correction`: **15.9% human vs 49.7% synthetic**, AUC 0.669, after two rounds of
hand-audit (dates like "el primero de septiembre" and card statements like "mi tarjeta la
que termina en 5510" removed; cue must sit next to a run of ≤ 2 digits).
**Mechanism:** the model holds the exact string in its context and can diff it.

### F6. The synthetic caller is more polite and attentive than a person on a service call
| feature | human | synthetic |
|---|---|---|
| `closing_ritual` — full goodbye in the last 3 turns | 4.4% | 41.4% |
| `calls_agent_by_name` — uses the agent's name, said once in the greeting | 10.6% | 45.0% |
| `one_word_negation` — turns that are a single word | 10.5% | 2.7% |

---

## WEAK — measured, kept out

- `grants_permission_on_interrupt` and `checks_line_alive`: 0/113 humans vs 16/169
  synthetic. Perfectly one-sided (Fisher p≈0.0006) but too rare to carry weight.
- `contrast_correction` ("X, no Y"): 0.9% vs 13.6%, AUC 0.564. Same story.
- The forced choice ("¿cuenta A o crédito B?"): 7-point gap. Nothing.

---

## RETRACTED

- **"One of the two products does not exist."** Invented. The agent introduces both
  products unprompted (195/216, 206/215), but 21 and 9 callers raise them first. No
  catalogue to check. Renamed `forced_choice`.
- **"The digit trap is rare (19%)."** That was the phrase "es correcto"; the read-back
  itself is in 81% / 52% of calls (F2).
- **"A human corrects the wrong digit; a language model confirms it."** Both groups accept
  the wrong read-back at the same low rate. See F5 for what actually differs.
- **"Whisper may be inflating the synthetic word counts."** See F3 — the agent channel
  disproves it. This worry drove the original "safe features" split; that split is now
  just a compact variant.
- **"The synthetic caller offers to repeat (30% vs 3.5%)."** The regex scored on *con eso*
  ("no, con eso está bien" — declining help). Without it: null. Dropped.
- **"Human and synthetic callers have the same talk time and turn count."** Whisper-span
  artefact. VAD says 21.9 vs 15.0 turns.

---

## Open

- Behavioral view fold leakage: fusion train OOF 0.992 vs val 0.960; semantic shows no gap.
- A better VAD lifts `speech_rate` (0.833 → 0.885 with dataset-quality turns).
- Persona overlap train↔val (19 of 25 groups): val understates the drop on the hidden set.
- The brief says the agent "asks about things that do not exist". Not located in the script.
