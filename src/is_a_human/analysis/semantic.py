"""Semantic head: caller transcript -> twelve deterministic measurements.

Ported from the semantic layer on branch max-semantic (src/semantic/features.py),
where every measurement below was found by a blind A/B read of 226 calls and then
kept only if it separated the classes on train (|AUC - 0.5| >= 0.10). Nothing here
is fitted; it is string matching and counting over the caller's turns.

What it measures, in one line: the synthetic caller packs more words into each
second of speech, corrects a misread digit by naming its position, performs the
full goodbye, and uses the agent's name. The human answers "No." and hangs up.

Inputs are Whisper turns [{channel, start, end, text}] plus the seconds the caller
actually spoke according to the VAD ledger -- Whisper's own segment spans include
the silence around speech and cannot be a rate denominator.

Every feature here came out of a blind A/B read of 226 train calls (the reviewer
was not told which group was which). Findings are numbered as in that report.
Nothing in here is fitted; these are plain measurements over the text.

Convention: features are named so that the *direction* is readable. Values are
rates or counts per call, never raw word counts, since absolute word counts
track how well the ASR heard the channel rather than how much was said.
"""
from __future__ import annotations

import re
import unicodedata

CALLER, AGENT = 0, 1


def norm(s: str) -> str:
    """Lowercase, strip accents, collapse spaces. ASR accents are unreliable."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


# ---- phrase inventories -------------------------------------------------
# §5  the caller names WHERE the digit is wrong instead of just rejecting it
# Tightened after a hand-audit of 14 matches: "el primero" was catching dates
# ("el primero de septiembre") and a bare "termina en" was catching the caller
# echoing their own card number. Now a digit must appear within ~20 chars.
_POS_CUE = (r"termina (?:en|con)|al final|ultim[oa]s?|primer(?:os?)? digitos?|penultim|"
            r"en lugar de|en vez de|despues del|antes del|de en medio|el que sigue")
# A correction names a position and one or two digits ("termina en 06", "el
# ultimo es 5"). A card statement names four ("termina en 5510"). Requiring a
# run of at most two digits next to the cue separates them.
_SHORT_RUN = r"(?<!\d)\d{1,2}(?!\d)"
POSITIONAL = re.compile(
    rf"(?:{_POS_CUE})[^.?!]{{0,20}}{_SHORT_RUN}|{_SHORT_RUN}[^.?!]{{0,20}}(?:{_POS_CUE})",
    re.I)
# "mi tarjeta la que termina en 5510" / "la de credito, la que termina en 2246":
# the caller stating their own card. Third audit pass -- the earlier guard
# required "tarjeta" directly before "termina" and missed every one of these.
POS_NOT_A_CORRECTION = re.compile(r"(?:tarjeta|cuenta)\b[^.?!]{0,25}termina", re.I)
# "en lugar del 10 de septiembre", "despues del 20 de agosto": a date, not a
# digit position. A turn that names a month is not correcting a reference.
POS_DATE = re.compile(r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
                      r"septiembre|setiembre|octubre|noviembre|diciembre)\b", re.I)
# §5  explicit "X, not Y" contrast
CONTRAST = re.compile(r",\s*no\s+\d|\bes\b[^.?!]{0,15}\bno\b\s+\d", re.I)

# §15 checking the line is alive
LINE_ALIVE = re.compile(r"\bbueno\b\s*\?|sigue (ahi|alli)|me escucha|\bhola\b\s*\?", re.I)

# §13 granting permission after being interrupted
PERMISSION = re.compile(r"\badelante\b|sin problema|no hay problema|no se preocupe|"
                        r"claro que si|por supuesto|dime|digame|si claro", re.I)

# §24 offering to repeat / checking if more is needed
# Review finding: "con eso" carried this whole feature (41 of the matches) and
# almost always meant "no, con eso esta bien" -- declining help, the opposite
# of offering to repeat. Removed. What is left is the literal offer.
OFFERS = re.compile(r"(quiere|gusta|necesita|desea)[^.?!]{0,20}(repit|repet)|"
                    r"(lo|la|se lo) repito|\brepito\b", re.I)

# §6  a full closing ritual rather than "gracias"
CLOSING = re.compile(r"que teng[ao]\s+(?:buen|muy|excelente|bonit|linda|feliz)|"
                     r"buen dia|buena tarde|buenas tardes|hasta luego|"
                     r"le agradezco|muy amable|su atencion", re.I)

# §18 blunt pushback
PUSHBACK = re.compile(r"esta mal|equivocad|incorrect|lo estas diciendo mal", re.I)

# §10 addressing the agent
BY_NAME = re.compile(r"\bmarina\b", re.I)
BY_TITLE = re.compile(r"senorita", re.I)

# agent-side anchors
AG_INTERRUPT = re.compile(r"que (la|lo|le) interrump", re.I)
AG_TWO_OPTS = re.compile(r"nomina\s*plus.{0,40}credito verde|credito verde.{0,40}nomina\s*plus", re.I)
AG_ELABORATE = re.compile(r"platicar|cuenteme|me puede contar|un poquito mas|"
                          r"con mas detalle|que paso", re.I)

W2D = {"cero":"0","uno":"1","dos":"2","tres":"3","cuatro":"4","cinco":"5",
       "seis":"6","siete":"7","ocho":"8","nueve":"9"}


def digits(text: str) -> str:
    t = norm(text)
    for w, d in W2D.items():
        t = re.sub(rf"\b{w}\b", d, t)
    return re.sub(r"[^\d]", "", t)


def _next_caller(turns, i, n=2):
    """The caller's next n turns after index i, as one normalised string."""
    out = []
    for t in turns[i + 1:]:
        if t["channel"] == CALLER:
            out.append(t["text"])
            if len(out) >= n:
                break
    return norm(" ".join(out))


def extract(turns: list[dict], caller_speech_s: float | None = None) -> dict[str, float]:
    """turns: Whisper turns for the call. caller_speech_s: seconds the caller
    actually spoke, from the VAD. Whisper segment spans swallow the silence
    around speech (about 2.2x the true duration on this data), so they cannot
    be the denominator of a rate; the VAD can."""
    turns = sorted(turns, key=lambda x: x["start"])
    caller = [t for t in turns if t["channel"] == CALLER]
    agent = [t for t in turns if t["channel"] == AGENT]
    if not caller:
        return {k: 0.0 for k in FEATURE_NAMES}

    ctext = [norm(t["text"]) for t in caller]
    allc = " ".join(ctext)
    nc = len(caller)
    words = [len(t.split()) for t in ctext]

    f: dict[str, float] = {}

    # §4 bare vs framed turns ------------------------------------------------
    f["bare_turn_rate"] = sum(w <= 2 for w in words) / nc
    f["digits_only_turn_rate"] = sum(bool(re.fullmatch(r"[\d\s,.\-]+", t)) for t in ctext) / nc
    f["one_word_negation"] = sum(bool(re.fullmatch(r"no\.?|correcto\.?|exacto\.?|si\.?", t))
                                 for t in ctext) / nc
    f["median_turn_words"] = float(sorted(words)[len(words) // 2])
    f["long_turn_rate"] = sum(w >= 15 for w in words) / nc

    # §5 precision when correcting a number ----------------------------------
    positional = [t for t in ctext
                  if POSITIONAL.search(t) and not POS_NOT_A_CORRECTION.search(t)
                  and not POS_DATE.search(t)]
    f["positional_correction"] = float(bool(positional))
    f["contrast_correction"] = float(bool(CONTRAST.search(allc)))

    # §15 / §13 / §24 conversational repair rituals ---------------------------
    f["checks_line_alive"] = float(bool(LINE_ALIVE.search(allc)))
    f["offers_to_repeat"] = float(bool(OFFERS.search(allc)))
    grants = 0.0
    for i, t in enumerate(turns):
        if t["channel"] == AGENT and AG_INTERRUPT.search(norm(t["text"])):
            if PERMISSION.search(_next_caller(turns, i)):
                grants = 1.0
            break
    f["grants_permission_on_interrupt"] = grants

    # §6 closing ritual -------------------------------------------------------
    tail = " ".join(ctext[-3:])
    f["closing_ritual"] = float(bool(CLOSING.search(tail)))
    f["closing_words"] = float(len(tail.split()))

    # §12 / §23 elaboration ----------------------------------------------------
    two_opt_len = 0.0
    for i, t in enumerate(turns):
        if t["channel"] == AGENT and AG_TWO_OPTS.search(norm(t["text"])):
            two_opt_len = float(len(_next_caller(turns, i, 1).split()))
            break
    f["two_options_reply_words"] = two_opt_len
    elab = 0.0
    for i, t in enumerate(turns):
        if t["channel"] == AGENT and AG_ELABORATE.search(norm(t["text"])):
            elab = max(elab, float(len(_next_caller(turns, i, 1).split())))
    f["elaboration_words"] = elab
    f["max_turn_words"] = float(max(words))

    # §22 bundling name and reference in one turn ------------------------------
    f["bundles_name_and_ref"] = float(any(
        len(digits(t["text"])) >= 6 and len(norm(t["text"]).split()) >= 6 for t in caller))

    # §17 off-script questions --------------------------------------------------
    f["question_turn_rate"] = sum("?" in t["text"] for t in caller) / nc

    # §18 friction --------------------------------------------------------
    f["blunt_pushback"] = float(bool(PUSHBACK.search(allc)))

    # §10 how the agent is addressed --------------------------------------------
    f["calls_agent_by_name"] = float(bool(BY_NAME.search(allc)))
    f["calls_agent_by_title"] = float(bool(BY_TITLE.search(allc)))

    # speech rate ---------------------------------------------------------------
    # The strongest single signal in the layer: the synthetic caller packs far
    # more words into each second of speech (review: 2.76 vs 1.71 words/s of
    # VAD-measured speech, AUC 0.885 alone). The agent -- the same TTS voice on
    # every call -- shows no such gap, which rules out the ASR hearing the two
    # groups differently. Falls back to Whisper spans if no VAD seconds given.
    total_words = float(sum(words))
    if caller_speech_s is None:
        caller_speech_s = sum(t["end"] - t["start"] for t in caller)
    f["speech_rate"] = total_words / max(caller_speech_s, 1e-6)

    return f



# The twelve shipped measurements, strongest first (train AUC in the comment).
# Prefixed so they group cleanly in CallFeatures and stay out of the
# behavioural name list.
SEMANTIC_HEAD_FEATURES = (
    "sem_speech_rate",              # 0.83  words / s of VAD speech
    "sem_median_turn_words",        # 0.78
    "sem_one_word_negation",        # 0.22  (human-leaning)
    "sem_bare_turn_rate",           # 0.24  (human-leaning)
    "sem_digits_only_turn_rate",    # 0.26  (human-leaning)
    "sem_closing_words",            # 0.71
    "sem_long_turn_rate",           # 0.70
    "sem_closing_ritual",           # 0.69
    "sem_max_turn_words",           # 0.68
    "sem_calls_agent_by_name",      # 0.67
    "sem_positional_correction",    # 0.67
    "sem_question_turn_rate",       # 0.64
)
SEMANTIC_AVAILABLE = "semantic_available"
FEATURE_NAMES = SEMANTIC_HEAD_FEATURES

SEMANTIC_ZEROS = {name: 0.0 for name in SEMANTIC_HEAD_FEATURES}
SEMANTIC_ZEROS[SEMANTIC_AVAILABLE] = 0.0


def extract_semantic_features(
    turns: list[dict],
    caller_speech_s: float | None = None,
) -> dict[str, float]:
    """Whisper turns (+ VAD caller seconds) -> the twelve shipped measurements.

    Returns SEMANTIC_ZEROS with semantic_available=0 when there is no caller
    text, so a row without a transcript is representable and the fast path can
    tell the two apart.
    """
    if not turns or not any(t.get("channel") == CALLER and str(t.get("text") or "").strip()
                            for t in turns):
        return dict(SEMANTIC_ZEROS)
    raw = extract(turns, caller_speech_s=caller_speech_s)
    out = {f"sem_{k}": float(raw[k]) for k in (n[4:] for n in SEMANTIC_HEAD_FEATURES)}
    out[SEMANTIC_AVAILABLE] = 1.0
    return out
