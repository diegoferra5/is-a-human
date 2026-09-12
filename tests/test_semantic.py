"""Semantic probe features from transcript turns (no ASR in these tests)."""

import pytest

from is_a_human.analysis.semantic import digit_trap_pairs, extract_semantic_features

pytestmark = pytest.mark.semantic


def _turn(channel: int, start: float, end: float, text: str) -> dict:
    return {"channel": channel, "start": start, "end": end, "text": text}


def test_digit_trap_detects_one_digit_mutation():
    agent = [
        _turn(1, 0.0, 2.0, "su referencia es 12345678"),
        _turn(1, 8.0, 10.0, "confirma 12345670 por favor"),
    ]
    assert digit_trap_pairs(agent) == [("12345678", "12345670")]


def test_digit_trap_ignores_unrelated_numbers():
    agent = [
        _turn(1, 0.0, 2.0, "codigo 11111111"),
        _turn(1, 3.0, 4.0, "otro codigo 99999999"),
    ]
    assert digit_trap_pairs(agent) == []


def test_extract_semantic_features_objected_trap(report_metrics):
    turns = [
        _turn(1, 0.0, 2.0, "la referencia es 12345678"),
        _turn(0, 2.2, 3.0, "de acuerdo"),
        _turn(1, 8.0, 10.0, "entonces 12345670 es correcto?"),
        _turn(0, 10.2, 11.5, "no, eso es incorrecto"),
    ]
    features = extract_semantic_features(turns)
    report_metrics(**features)

    assert features["digit_trap_count"] == 1.0
    assert features["digit_trap_caller_objected"] == 1.0


def test_extract_semantic_features_bot_confirms_trap(report_metrics):
    turns = [
        _turn(1, 0.0, 2.0, "la referencia es 12345678"),
        _turn(1, 8.0, 10.0, "confirma 12345670"),
        _turn(0, 10.1, 11.0, "si, correcto"),
    ]
    features = extract_semantic_features(turns)
    report_metrics(**features)

    assert features["digit_trap_count"] == 1.0
    assert features["digit_trap_caller_objected"] == 0.0


def test_false_dilemma_and_announced_interrupt(report_metrics):
    turns = [
        _turn(1, 0.0, 2.0, "es sobre producto A o producto B?"),
        _turn(1, 2.5, 3.5, "perdon que la interrumpa"),
        _turn(0, 3.6, 4.5, "eh pues no se"),
    ]
    features = extract_semantic_features(turns)
    report_metrics(**features)

    assert features["false_dilemma_count"] == 1.0
    assert features["announced_interrupt_count"] == 1.0
    assert features["caller_filler_count"] >= 1.0


def test_empty_transcript_is_all_zeros():
    features = extract_semantic_features([])
    assert features == {
        "digit_trap_count": 0.0,
        "digit_trap_caller_objected": 0.0,
        "false_dilemma_count": 0.0,
        "announced_interrupt_count": 0.0,
        "caller_filler_count": 0.0,
    }
