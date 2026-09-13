"""Semantic features: one transcript -> one named feature dict.

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
POSITIONAL = re.compile(
    rf"(?:{_POS_CUE})[^.?!]{{0,20}}\d|\d[^.?!]{{0,20}}(?:{_POS_CUE})", re.I)
# "mi tarjeta termina en 5510" is the caller stating their own card, not correcting
# a misread. Second audit pass: these were the only remaining false positives.
POS_NOT_A_CORRECTION = re.compile(r"(?:tarjeta|cuenta)\s+(?:que\s+)?termina", re.I)
# §5  explicit "X, not Y" contrast
CONTRAST = re.compile(r"\bno\b[^.?!]{0,25}\bsino\b|,\s*no\s+\d|\bes\b[^.?!]{0,15}\bno\b\s+\d", re.I)

# §15 checking the line is alive
LINE_ALIVE = re.compile(r"\bbueno\b\s*\?|sigue (ahi|alli)|me escucha|me oye|"
                        r"\bhola\b\s*\?|esta ahi|sigues ahi|me copia", re.I)

# §13 granting permission after being interrupted
PERMISSION = re.compile(r"\badelante\b|sin problema|no hay problema|no se preocupe|"
                        r"claro que si|por supuesto|dime|digame|si claro", re.I)

# §24 offering to repeat / checking if more is needed
OFFERS = re.compile(r"(quiere|gusta|necesita|desea)[^.?!]{0,20}(repit|repet)|"
                    r"(lo|la|se lo) repito|repito|necesita algo mas|algo mas\s*\?|"
                    r"le sirve|es suficiente|con eso", re.I)

# §6  a full closing ritual rather than "gracias"
CLOSING = re.compile(r"que teng[ao]|buen dia|buena tarde|buenas tardes|hasta luego|"
                     r"le agradezco|muy amable|su atencion|excelente dia", re.I)

# §18 blunt pushback
PUSHBACK = re.compile(r"esta mal|equivocad|incorrect|lo estas diciendo mal|"
                      r"no es asi|ya se lo dije|ya le dije|otra vez", re.I)

# §19 noticing at the end that something was never given
MISSING = re.compile(r"no me (dio|dijo|ha dado|dieron)|nunca me|falta|no me lo", re.I)

# §10 addressing the agent
BY_NAME = re.compile(r"\bmarina\b", re.I)
BY_TITLE = re.compile(r"senorita|senora\b|senor\b(?! [a-z])", re.I)

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


def extract(turns: list[dict]) -> dict[str, float]:
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
                  if POSITIONAL.search(t) and not POS_NOT_A_CORRECTION.search(t)]
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

    # §18 / §19 friction --------------------------------------------------------
    f["blunt_pushback"] = float(bool(PUSHBACK.search(allc)))
    f["notices_missing"] = float(bool(MISSING.search(" ".join(ctext[-4:]))))

    # §10 how the agent is addressed --------------------------------------------
    f["calls_agent_by_name"] = float(bool(BY_NAME.search(allc)))
    f["calls_agent_by_title"] = float(bool(BY_TITLE.search(allc)))

    return f


# Features whose value depends on what the AGENT said, not only the caller.
# Everything else needs channel 0 alone -- which halves transcription time at
# serving, where we pay for Whisper live. Keep this in sync with extract().
AGENT_DEPENDENT = {
    "grants_permission_on_interrupt",
    "two_options_reply_words",
    "elaboration_words",
}


FEATURE_NAMES = [
    "bare_turn_rate", "digits_only_turn_rate", "one_word_negation",
    "median_turn_words", "long_turn_rate",
    "positional_correction", "contrast_correction",
    "checks_line_alive", "offers_to_repeat", "grants_permission_on_interrupt",
    "closing_ritual", "closing_words",
    "two_options_reply_words", "elaboration_words", "max_turn_words",
    "bundles_name_and_ref", "question_turn_rate",
    "blunt_pushback", "notices_missing",
    "calls_agent_by_name", "calls_agent_by_title",
]
