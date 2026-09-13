# Semantic layer — what we know, and how we know it

All numbers: **train split only, 282 calls**. Val (71) untouched.
Terminology: "the agent" = the bank's scripted AI, channel 1. "The caller" = the one we
classify, channel 0.

---

## CONFIRMED — measured on all 282 train calls

### F1. The agent runs the same script on every call
Identical opening line word-for-word in all 353 calls, same agent name, same question set.
Beat order varies in the middle and beats are often skipped, but `ask_ref` precedes
`confirm_ref` in 94% of calls where both fire.
**How:** `src/semantic/step0_script_check.py`, 282 calls.
**So:** deterministic phrase matching will generalize. Never match on position.

### F2. The traps fire more often on human callers
| beat | human | synthetic |
|---|---|---|
| forced choice | 85.8% | 71.0% |
| confirm reference | 87.6% | 68.6% |
| altered read-back | 81.4% | 52.1% |

Not explained by call duration — the gap survives inside every length quartile.
**How:** same script; length control by duration quartile.
**So:** *"did the trap fire"* can never be a feature. It leaks the label without measuring
the caller. Every feature must be conditioned on the trap having fired, and carry its own
denominator.

### F3. Synthetic callers produce 54% more words in the same talk time
Same call length, same turn count (14), same caller talk time (88 s), but 107 words vs 70.
The agent gets ~4.5 fewer turns and 52 fewer words — which is *why* F2 happens.
**How:** word and duration counts over 282 calls.
**CAUTION:** may be an ASR artifact. Whisper transcribes clean synthetic audio more
completely than noisy human phone audio. Identical talk *time* with different word counts
is exactly what an ASR bias would look like. **Do not ship a word-count feature until it is
checked against audio-derived durations.**

### F4. The altered read-back separates the two groups — in the opposite direction
The agent states the reference number, then later repeats it altered by one digit.
Of the calls where this fires (92 human / 88 synthetic):

| caller reaction | human | synthetic | gap |
|---|---|---|---|
| confirms the wrong number | 35.9% | 20.5% | **+15.4** |
| objects / corrects | 22.8% | 37.5% | **-14.7** |
| restates a number | 12.0% | 8.0% | +4.0 |
| other | 28.3% | 33.0% | -4.7 |
| silent | 1.1% | 1.1% | 0.0 |

**The human accepts the wrong number. The synthetic catches it.**
Mechanism: the LLM holds the number in its context window with perfect recall. The human
invented that number two minutes ago, is on a phone, and is not tracking it.
**Why this should generalize:** it detects the caller for being *too good*, not for being
clumsy. Perfect context recall is a property of every LLM, independent of which TTS or
which model — so it should survive unseen engines.
**How:** digit runs >=6 extracted from agent turns, first vs later, near-miss = altered;
caller's next 2 turns classified by regex. 180 calls.
**NOT YET VERIFIED:** the "objects" regex includes `\bno\b`, very common in Spanish, so
part of that 37.5% may be noise. 30% falls into "other", meaning the rules are crude.
Needs a hand-audit of ~30 events before this number is quotable.

---

## WEAK — measured, but too small to carry anything

### F5. The forced choice barely separates
"¿Es sobre su cuenta A o su B?" — fired in 94 human / 117 synthetic calls.
Largest single gap 7.1 points (picks one: 48.9% vs 41.9%).
Combining hedge + both + neither into one "did not commit" feature: **5.4% human vs 16.2%
synthetic**, a 3x ratio but only 11 human / 19 synthetic calls.
**So:** keep as one small feature, not as a centerpiece. 45% of answers landed in
"off-topic" for both groups, which means the rules are bad, not that callers are off topic.

---

## DISPROVEN / RETRACTED

### R1. "One of the two products in the forced choice does not exist"
No evidence. Invented. The agent introduces both products unprompted (195/216 and 206/215
calls), but 21 and 9 callers respectively raise them first, so some personas do hold them.
We have no product catalogue and cannot check.
**Renamed** to `forced_choice`. Detection value does not depend on the products being fake
— it depends on how the caller handles a question that presupposes something about them.

### R2. "The digit trap is too rare to use (19% of calls)"
Wrong measurement. 19% was the rate of the literal phrase "es correcto". The altered
read-back itself occurs in **81% of human and 52% of synthetic calls** — see F4.

### R3. "A human corrects the wrong digit, an LLM agreeably confirms it"
The stated hypothesis for this whole layer. **Measured and reversed** — see F4.

---

## Open questions

- Is F3 real behaviour or Whisper bias? Check against audio-derived turn durations.
- What is in the 30% "other" bucket at the read-back, and the 45% "off-topic" bucket at the
  forced choice? Those are the largest unexplained groups in both probes.
- The brief says the agent "asks about things that do not exist." We have not located that
  moment in the script. The forced choice was a guess and it did not hold up.
- Does F4 survive a hand-audit of 30 events?
