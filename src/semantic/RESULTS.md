# Semantic layer — measured results

Working notes for layer 2. Numbers are **train split, 282 calls** (113 human, 169
synthetic) unless a row says val. Val (71 calls) has been scored twice — once by an
independent review, once after the fixes it prompted — and is now spent: see "Val".

Reproduce: `python -m src.semantic.evaluate` (per-feature) · `python -m src.semantic.train`
(model + CV) · `python -m src.train_all` (fusion with the behavioral view).

---

## How we got here

1. `step0_script_check.py` — the agent runs the same script on every call, and reaches its
   later beats more often on human callers (see `analysis/step0-findings.md`).
2. `build_blind_set.py` — 226 balanced transcripts, labelled only "group A" / "group B",
   sponsor name scrubbed, read by an independent session that did not know which was which.
   It produced 37 observations; we measured the 14 that looked like properties of *how a
   language model behaves* rather than of how this particular stack was configured.
3. An **adversarial review** of the resulting code and numbers (14 findings). The important
   ones: the layer's main signal is speech density, not the precision/politeness story we
   had told; one of the features measured the wrong phrase; the ASR-artifact worry that had
   shaped the feature list was wrong. All fixed below, and the retracted claims are listed.

Twice now, what we believed before measuring was wrong. Everything in this file is what
survived a measurement, and the section at the bottom says what did not.

---

## The 12 features that ship

Raw AUC: > 0.5 means the synthetic group scores higher, < 0.5 the human group. A
label-permuted feature lands in 0.44–0.56 (95% band), so anything inside that is nothing.

| feature | unit | human | synthetic | AUC |
| --- | --- | --- | --- | --- |
| **`speech_rate`** | words / s of VAD speech | 1.51 | 2.48 | **0.833** |
| `median_turn_words` | words | 4.0 | 6.4 | 0.784 |
| `one_word_negation` | share of turns | 10.5% | 2.7% | 0.224 |
| `bare_turn_rate` | share of turns | 38.8% | 23.3% | 0.238 |
| `digits_only_turn_rate` | share of turns | 11.1% | 3.3% | 0.259 |
| `closing_words` | words | 8.4 | 12.8 | 0.714 |
| `long_turn_rate` | share of turns | 7.7% | 15.0% | 0.702 |
| `closing_ritual` | share of calls | 4.4% | 41.4% | 0.685 |
| `max_turn_words` | words | 16.5 | 20.5 | 0.682 |
| `calls_agent_by_name` | share of calls | 10.6% | 45.0% | 0.672 |
| `positional_correction` | share of calls | 15.9% | 49.7% | 0.669 |
| `question_turn_rate` | share of turns | 4.7% | 7.8% | 0.635 |

`speech_rate` divides the caller's Whisper word count by the seconds the caller actually
spoke according to the shipped VAD (`Call.turns()`, the same one the behavioral view uses,
so it exists at judging time). Whisper's own segment spans cannot be the denominator: they
swallow the silence around speech and run about 2.2× the true duration on this data.

### The model

`StandardScaler` → `LogisticRegression(class_weight="balanced", C=0.1)`. Twelve weights and
an intercept. Logistic regression because 282 rows is too few for a tree ensemble not to
memorise speakers, and because a linear model can print *why* a call scored high
(`SemanticView.explain`). `balanced` so the model does not learn train's 60/40 class ratio
as a standing lean toward "synthetic" (val is 48/52; the judging set's ratio is unknown).
`C=0.1` because several features are correlated (|r| up to 0.70) and at the default `C=1`
three small weights had undetermined sign under bootstrap.

### Cross-validation on train (5-fold, mean of 20 seeds)

| feature set | n | AUC | acc |
| --- | --- | --- | --- |
| all 12 | 12 | **0.914** (sd 0.006) | 84.6% |
| compact (`SAFE_FEATURES`) | 5 | 0.904 (sd 0.005) | 82.6% |
| `speech_rate` alone | 1 | 0.833 (sd 0.002) | 80.1% |

Selection bias from choosing features on train: re-running the screen inside each fold
gives +0.001. Speaker leakage inside train: a persona-grouped CV (built from the invented
names that recur in the transcripts) gives the same AUC as random folds, to three decimals.

### Val — scored once after the fixes

71 calls, 37 human / 34 synthetic, majority baseline 52.1%. Confusion as
`[[hum→hum, hum→syn], [syn→hum, syn→syn]]`.

| model | AUC | acc | confusion |
| --- | --- | --- | --- |
| semantic, 12 features | **0.897** | 83.1% | `[[32, 5], [7, 27]]` |
| semantic, 5 compact | 0.890 | 84.5% | `[[32, 5], [6, 28]]` |
| **fusion: behavioral + semantic** | **0.960** | **88.7%** | `[[31, 6], [2, 32]]` |
| behavioral alone (context) | 0.923 | 85.9% | `[[31, 6], [4, 30]]` |

**This budget is spent.** Val was scored by the review (old model: 0.891 / 81.7%) and once
more here. Any further change chosen by looking at these numbers invalidates them.

Two cautions on reading val as "the hidden set":

- 19 of the 25 train persona groups also appear in val. The splits are speaker-disjoint by
  *voice*, but scenarios, invented names and reference numbers are shared. The judging set
  shares none of that. Expect somewhat below these numbers there.
- The fusion's train OOF is 0.992 but its val is 0.960. The semantic view shows no such gap
  (0.914 → 0.897), so the difference is likely the behavioral view leaking speakers across
  random folds inside train. Not checked; not this layer's file.

---

## What the layer actually measures

**Mostly: the synthetic caller talks denser.** 2.5 words per second of speech against 1.5.
The agent — the same TTS voice on every call — has identical words-per-second in both
groups, which rules out the ASR hearing the two groups differently. This is real caller
behaviour and it is the layer's backbone: one number reaches 0.833 on its own.

**The rest is the increment.** With speech rate controlled, the remaining features add
about +0.08 AUC jointly, and no single one adds more than +0.02. They are worth having, and
they are the part a judge can *see* in a transcript:

- `positional_correction` — the agent misreads the reference number; the synthetic caller
  says *where* ("termina en 06, no 05"), the human says "está mal" and re-dictates. The
  model holds the exact string in context and can diff it. 15.9% vs 49.7%.
- `closing_ritual` — the full goodbye ("que tenga excelente día") instead of "gracias".
  4.4% vs 41.4%.
- `calls_agent_by_name` — the agent says her name once, in the greeting; the synthetic
  caller uses it, the human says "señorita". 10.6% vs 45.0%.
- the terseness family (`one_word_negation`, `bare_turn_rate`, `digits_only_turn_rate`) —
  a person answers "No." A language model answers in a sentence.

**The line for the judges:** the synthetic caller packs more words into every second of
speech, and on top of that it is more precise, more polite and more attentive than a
person on a service call has any reason to be.

---

## Measured and rejected

| feature | human | synthetic | AUC | why |
| --- | --- | --- | --- | --- |
| `bundles_name_and_ref` | 62.8% | 77.5% | 0.573 | weak |
| `contrast_correction` | 0.9% | 13.6% | 0.564 | right direction, too rare |
| `two_options_reply_words` | 3.5 | 5.4 | 0.559 | null band |
| `grants_permission_on_interrupt` | 0.0% | 9.5% | 0.547 | null band by AUC; 0/113 vs 16/169 is real (Fisher p≈0.0006) but too rare to use |
| `checks_line_alive` | 0.0% | 9.5% | 0.547 | same |
| `offers_to_repeat` | 2.7% | 10.1% | 0.537 | null band — see retracted |
| `calls_agent_by_title` | 11.5% | 3.6% | 0.460 | null band |
| `blunt_pushback` | 11.5% | 2.4% | 0.454 | null band |
| `elaboration_words` | 9.5 | 10.3 | 0.477 | null band |

---

## Retracted — claims that did not survive measurement

- **"One of the two products the agent offers does not exist."** Invented; no catalogue to
  check. Renamed `forced_choice`; it separates by 7 points, i.e. nothing.
- **"The digit trap is rare (19% of calls)."** Wrong measurement — that was the literal
  phrase "es correcto". The altered read-back occurs in 81% of human and 52% of synthetic
  calls.
- **"A human corrects the wrong digit; a language model agreeably confirms it."** The
  founding hypothesis of the layer. Measured: both groups accept the wrong read-back at the
  same low rate. What differs is *how precisely* they correct — `positional_correction`.
- **"Whisper hears clean synthetic audio better, so word-count features may be measuring
  the ASR."** Wrong. The agent channel is the control: same TTS voice everywhere, identical
  words-per-second across groups. The 54% word gap is caller behaviour. This worry was the
  entire justification for the "safe features" split; that list is now just a compact
  variant.
- **"The synthetic caller offers to repeat information nobody asked for (30% vs 3.5%)."**
  The regex behind that number scored almost entirely on the phrase *con eso*, which in
  these calls is "no, con eso está bien" — declining help. With it removed the feature is
  null. Dropped.
- **"We detect the model by remembering everything, correcting too precisely, and saying
  goodbye too well."** Overstated. Those features fall to AUC 0.51–0.53 once speech rate is
  controlled. They are the increment, not the mechanism.

---

## Known risks

1. **Speech density could move.** It is a property of LLM+TTS stacks generally and should
   transfer, but a terser system prompt or a slower voice would move it a long way, and
   the layer's second leg (+0.08) is thin.
2. **`speech_rate` depends on the VAD.** The behavioral view's energy VAD is noisier than
   the dataset's own turn files (0.833 vs 0.885 with ground-truth turns). It is the one
   that exists at judging, so it is the honest number; a better VAD lifts this feature
   for free.
3. **`question_turn_rate` reads Whisper's punctuation**, which the decoder predicts from
   prosody. It is the one feature that *survives* speech-rate control (residual AUC 0.72),
   so it stays — but it is not ASR-independent and should not be described as such.
4. **Persona overlap between train and val** (above). Val is not a clean proxy for judging.
5. **Three small weights contradict their marginal direction** (`bare_turn_rate`,
   `closing_words`, `max_turn_words`, all |w| ≤ 0.05). The prediction uses them;
   `explain()` hides them so a reader is never shown "many short turns → synthetic".
6. **Latency.** Whisper runs live on the caller channel only: 8–9 s for a 130–170 s call on
   an M2, ~15× realtime. Fine for judging.

---

## Review log

Findings from the adversarial review and what was done (`analysis/facts.md` has the
measurement detail):

| # | finding | action |
| --- | --- | --- |
| 1 | layer ≈ speech rate; ASR-artifact worry wrong | `speech_rate` added; story rewritten |
| 2 | `offers_to_repeat` measured "con eso" | phrase removed; feature dropped (null) |
| 3 | view not wired; `serve.py` could not start | wired into `serve.py` / `train_all.py`; fusion trained |
| 4 | `positional_correction` guard missed "la que termina en 5510" | guard loosened; short-digit-run required |
| 5 | serving wrote into `transcripts/`; cache not cleared on `load()` | `out_dir=None` on live path; cache keyed by feature set and cleared |
| 6 | three weights with undetermined sign; `explain()` would mislead | `C=0.1`; `explain()` filters sign-contradicting features |
| 7 | `facts.md` talk-time numbers from Whisper spans | recomputed from VAD |
| 8 | `notices_missing` dead | removed |
| 9 | ~15 dead regex alternatives | removed |
| 10 | `evaluate.py` reported `max(auc, 1-auc)` | raw AUC with null band |
| 11 | `--safe` overwrote `semantic.pkl` | saves to `semantic_safe.pkl` |
| 12 | docs out of date | this file |
| 13 | `CLOSING` matched "que tengo" | tightened |
| — | `train_all.py` folded over train+val | filtered to train (one line; the team's file) |

Checked and clean: no crash paths, no order/float nondeterminism, no shipped feature reads
the agent channel, feature order consistent across `fit` / `proba` / `explain` / `load`,
accent handling correct.

---

## Next

- [ ] Check the behavioral view for speaker leakage in its random folds (fusion 0.992 train
      OOF vs 0.960 val).
- [ ] Try the dataset-quality VAD (Silero) as the `speech_rate` denominator — measured
      +0.05 on that feature alone with ground-truth turns.
- [ ] Add a smoke test: POST a WAV, assert 200. This branch has no tests.
