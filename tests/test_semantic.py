"""Semantic head: transcript turns -> twelve deterministic measurements (no ASR here)."""

import pytest

from is_a_human.analysis.features import CallFeatures, behavioral_feature_names
from is_a_human.analysis.semantic import (
    SEMANTIC_AVAILABLE,
    SEMANTIC_HEAD_FEATURES,
    SEMANTIC_ZEROS,
    extract_semantic_features,
)

pytestmark = pytest.mark.semantic


def _turn(channel: int, start: float, end: float, text: str) -> dict:
    return {"channel": channel, "start": start, "end": end, "text": text}


def test_empty_transcript_is_zeros_and_unavailable():
    assert extract_semantic_features([]) == SEMANTIC_ZEROS
    assert extract_semantic_features([])[SEMANTIC_AVAILABLE] == 0.0


def test_agent_only_transcript_is_unavailable():
    turns = [_turn(1, 0.0, 2.0, "Buen día, le atiende Marina.")]
    assert extract_semantic_features(turns)[SEMANTIC_AVAILABLE] == 0.0


def test_feature_names_exist_on_callfeatures_and_stay_out_of_behavioral():
    fields = set(CallFeatures.__dataclass_fields__)
    assert set(SEMANTIC_HEAD_FEATURES) <= fields
    assert SEMANTIC_AVAILABLE in fields
    assert not (set(SEMANTIC_HEAD_FEATURES) & set(behavioral_feature_names()))


def test_speech_rate_uses_vad_seconds_not_whisper_spans():
    turns = [_turn(0, 0.0, 10.0, "uno dos tres cuatro cinco seis siete ocho nueve diez")]
    fast = extract_semantic_features(turns, caller_speech_s=4.0)
    slow = extract_semantic_features(turns, caller_speech_s=10.0)
    assert fast["sem_speech_rate"] == pytest.approx(10 / 4.0)
    assert slow["sem_speech_rate"] == pytest.approx(10 / 10.0)


def test_positional_correction_fires_on_a_named_digit():
    turns = [
        _turn(1, 0.0, 2.0, "Entonces me confirma 2284-5507. ¿Es correcto?"),
        _turn(0, 2.2, 4.0, "No, termina en 06. O sea, 2284-5506."),
    ]
    assert extract_semantic_features(turns, caller_speech_s=2.0)["sem_positional_correction"] == 1.0


def test_positional_correction_ignores_card_statements_and_dates():
    card = [_turn(0, 0.0, 3.0, "Es de mi tarjeta la que termina en 5510.")]
    date = [_turn(0, 0.0, 3.0, "Hice el pago el primero de septiembre en lugar del 10.")]
    assert extract_semantic_features(card, caller_speech_s=3.0)["sem_positional_correction"] == 0.0
    assert extract_semantic_features(date, caller_speech_s=3.0)["sem_positional_correction"] == 0.0


def test_closing_ritual_and_agent_name():
    ritual = [
        _turn(0, 0.0, 2.0, "Hola Marina, tengo un problema con mi cuenta."),
        _turn(0, 5.0, 8.0, "Muchas gracias por su apoyo, que tenga excelente día."),
    ]
    terse = [_turn(0, 0.0, 1.0, "Sí."), _turn(0, 5.0, 5.5, "Gracias.")]
    r = extract_semantic_features(ritual, caller_speech_s=5.0)
    t = extract_semantic_features(terse, caller_speech_s=1.5)
    assert r["sem_closing_ritual"] == 1.0 and r["sem_calls_agent_by_name"] == 1.0
    assert t["sem_closing_ritual"] == 0.0 and t["sem_calls_agent_by_name"] == 0.0
    assert t["sem_one_word_negation"] > r["sem_one_word_negation"]


def test_closing_ritual_does_not_match_que_tengo():
    turns = [_turn(0, 0.0, 2.0, "El número que tengo es 4470 8193."), _turn(0, 3.0, 4.0, "Gracias.")]
    assert extract_semantic_features(turns, caller_speech_s=3.0)["sem_closing_ritual"] == 0.0


def test_deterministic():
    turns = [
        _turn(1, 0.0, 2.0, "¿Me puede dar su nombre?"),
        _turn(0, 2.2, 4.0, "Sí, claro. Es Héctor Navarro."),
        _turn(0, 9.0, 10.0, "No, termina en 3."),
    ]
    a = extract_semantic_features(turns, caller_speech_s=3.0)
    b = extract_semantic_features(list(reversed(turns)), caller_speech_s=3.0)
    assert a == b
