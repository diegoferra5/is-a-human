# Semantic layer — measured results

Working notes for layer 2. Everything below is **train split, 282 calls** (113 human,
169 synthetic). Val (71 calls) is untouched until the very end.

Reproduce: `python -m src.semantic.evaluate` → `analysis/feature_scores.csv`

---

## How we got here

1. `step0_script_check.py` — confirmed the agent runs the same script on every call, and
   found that the script's traps fire more often on human callers (see `analysis/step0-findings.md`).
2. `build_blind_set.py` — cut 226 balanced transcripts into a folder labelled only "group A"
   and "group B", sponsor name scrubbed, and had an independent session read them without
   knowing which group was which.
3. That session produced 37 observations. We kept the 14 whose mechanism looked like a
   property of *how a language model behaves* rather than of how this particular stack was
   configured, and measured them here.

The blindness mattered. Before it we had three hypotheses, and all three were wrong when
measured — see "Retracted" at the bottom.

---

## The 12 features that survived (AUC >= 0.60)

| feature | human | synthetic | AUC | direction |
| --- | --- | --- | --- | --- |
| `median_turn_words` | 4.0 | 6.4 | 0.784 | synthetic longer |
| `one_word_negation` | 10.5% | 2.7% | 0.776 | human |
| `bare_turn_rate` | 38.8% | 23.3% | 0.762 | human |
| `digits_only_turn_rate` | 11.1% | 3.3% | 0.741 | human |
| `closing_words` | 8.4 | 12.8 | 0.714 | synthetic |
| `long_turn_rate` | 7.7% | 15.0% | 0.702 | synthetic |
| **`positional_correction`** | **17.7%** | **54.4%** | **0.684** | synthetic |
| `closing_ritual` | 6.2% | 42.6% | 0.682 | synthetic |
| `max_turn_words` | 16.5 | 20.5 | 0.682 | synthetic |
| **`calls_agent_by_name`** | **10.6%** | **45.0%** | 0.672 | synthetic |
| `question_turn_rate` | 4.7% | 7.8% | 0.635 | synthetic |
| `offers_to_repeat` | 3.5% | 30.2% | 0.633 | synthetic |

### Combined (5-fold CV on train)

| feature set | n | AUC | acc |
| --- | --- | --- | --- |
| all 12 | 12 | 0.898 | 80.8% |
| **without any length feature** | **5** | **0.860** | **78.4% |
| only length features | 6 | 0.808 | 73.1% |

**The second row is the one that matters.** Six of the twelve measure length, and Whisper
transcribes clean synthetic audio more completely than a noisy human on a phone line, so
those six are partly measuring the ASR. The five that do not count words still reach 0.857
on their own: `positional_correction`, `closing_ritual`, `calls_agent_by_name`,
`offers_to_repeat`, `question_turn_rate`.

---

## The four mechanisms

**1. The human is terse; the model cannot be.** A person on the phone answers "No." A
language model is trained to answer completely, so it says "No, that is not the right
number." Structural, not stylistic. *(6 length features — ASR-contaminated, see above.)*

**2. The model corrects with surgical precision.** The agent misreads the reference number.
Both notice. The human says "that's wrong" and re-dictates the whole thing; the model says
"it ends in 06, not 05." The model holds the exact string in its context and can diff it
positionally. **Perfect context recall belongs to every LLM, not to this one — this is the
feature most likely to survive unseen engines.**

**3. The model is polite by ritual.** It completes the closing ceremony (43% vs 6%) and
offers to repeat information nobody asked it to repeat (30% vs 3.5%). The human has solved
their problem and hangs up.

**4. The model was paying attention from second zero.** The agent says her name once, in the
greeting, while the human is still framing their complaint. The human never registers it and
says "señorita"; the model uses "Marina" (45% vs 11%). Same mechanism as #2.

**The line for the judges:** we do not detect the model by how it sounds. We detect it by
remembering everything, correcting too precisely, and saying goodbye too well.

---

## Measured and rejected

| feature | human | synthetic | AUC | why it failed |
| --- | --- | --- | --- | --- |
| `bundles_name_and_ref` | 62.8% | 77.5% | 0.573 | weak |
| `contrast_correction` | 0.9% | 13.6% | 0.564 | right direction, too rare |
| `two_options_reply_words` | 3.5 | 5.4 | 0.559 | weak |
| `grants_permission_on_interrupt` | 0.0% | 9.5% | 0.547 | 0 vs 16 calls, too rare |
| `checks_line_alive` | 0.0% | 9.5% | 0.547 | same |
| `calls_agent_by_title` | 11.5% | 3.6% | 0.540 | weak |
| `blunt_pushback` | 12.4% | 5.9% | 0.532 | weak |
| `elaboration_words` | 9.5 | 10.3 | 0.523 | nothing |
| `notices_missing` | 0.0% | 0.0% | 0.500 | never fires; regex is probably wrong |

`grants_permission_on_interrupt` and `checks_line_alive` are perfectly clean (no human ever
does either) but fire in only 16 of 169 synthetic calls. Kept out of the model; worth a
mention as evidence, not as a feature.

---

## Retracted — hypotheses that did not survive measurement

- **"One of the two products offered does not exist."** Invented. We have no product
  catalogue and cannot check. Renamed `forced_choice`; detection never depended on it.
- **"The digit trap is rare (19% of calls)."** Wrong measurement — that was the rate of the
  literal phrase "es correcto". The altered read-back itself occurs far more often.
- **"A human corrects the wrong digit; a language model agreeably confirms it."** This was
  the founding hypothesis of the whole layer. Measured: **both groups accept the wrong
  read-back at the same low rate.** What actually differs is *how precisely* they correct it
  — which became `positional_correction`.
- **"The false dilemma separates the groups."** 7-point gap. Effectively nothing.

---

## Known risks

1. **Length features are partly ASR.** Caller talk *time* is nearly identical between groups
   (87.6 s vs 88.4 s) while word counts differ by 54%. That is what an ASR bias looks like.
   The five length-free features are the safe core.
2. ~~`positional_correction` has not been hand-audited.~~ **Done.** Two rounds, 30 matches
   read against the agent turn that preceded them. The first version had ~14% false
   positives: `el primero` was matching dates ("el primero de septiembre") and a bare
   "termina en" was matching callers stating their own card number ("mi tarjeta termina en
   5510"). Tightened to require a digit within 20 characters of the positional cue, and to
   exclude "tarjeta/cuenta ... termina". Matches fell 173 -> 128; the feature's own AUC fell
   0.702 -> 0.684 but the class separation improved from 2.7x to 3.1x (17.7% vs 54.4%), and
   the full model went **up**, 0.892 -> 0.898. What was removed was noise, not signal.
3. **Trap fire rates leak the label** (`analysis/step0-findings.md`): the agent reaches the
   later traps in 81% of human calls but only 52% of synthetic ones, because synthetic
   callers crowd it out. Nothing here uses "did the trap fire" as a feature, and nothing
   should.
4. **No speaker-disjoint CV within train.** The 5-fold above may put the same caller in both
   folds. The train/val split is speaker-disjoint by design, so the val score at the end is
   the honest one.

---

## Next

- [x] Hand-audit `positional_correction` — done, see Known risks 2.
- [ ] Measure end-to-end latency: judging sends a WAV, so Whisper runs at inference time.
- [ ] Check the length features against audio-derived turn durations to separate real
      verbosity from ASR bias.
- [ ] Wire `features.extract()` into `src/views/semantic_view.py` (`_features` currently
      returns zeros).
- [ ] Score on val **once**, at the end.
