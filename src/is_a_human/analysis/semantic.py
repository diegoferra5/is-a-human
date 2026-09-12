"""Transcript probe features from agent/caller turn text.

No model. String matching on the agent's scripted traps and the caller's reply.
"""

from __future__ import annotations

import re
from typing import Iterable

_DIGITS = re.compile(r"\d{4,}")
_FILLERS = re.compile(r"\b(eh+|este|pues|mmm+|ah+)\b", re.IGNORECASE)
_OBJECTION = re.compile(
    r"\b(no|incorrect[oa]|fals[oa]|espera|momento|equivocad[oa])\b",
    re.IGNORECASE,
)
_INTERRUPT = re.compile(r"interrump", re.IGNORECASE)
_DILEMMA = re.compile(r"\bo\b", re.IGNORECASE)

FEATURE_NAMES = (
    "digit_trap_count",
    "digit_trap_caller_objected",
    "false_dilemma_count",
    "announced_interrupt_count",
    "caller_filler_count",
)


def _text(turn: dict) -> str:
    return str(turn.get("text") or "")


def _channel_turns(turns: Iterable[dict], channel: int) -> list[dict]:
    return sorted(
        (turn for turn in turns if turn.get("channel") == channel),
        key=lambda turn: float(turn.get("start") or 0.0),
    )


def _hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        return -1
    return sum(a != b for a, b in zip(left, right))


def digit_trap_pairs(agent_turns: list[dict]) -> list[tuple[str, str]]:
    """Numbers the agent restates later with exactly one digit changed."""
    numbers: list[str] = []
    for turn in agent_turns:
        numbers.extend(_DIGITS.findall(_text(turn)))

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, original in enumerate(numbers):
        for mutated in numbers[index + 1 :]:
            key = tuple(sorted((original, mutated)))
            if key in seen or _hamming(original, mutated) != 1:
                continue
            seen.add(key)
            pairs.append((original, mutated))
    return pairs


def caller_reply_after(turns: list[dict], agent_turn: dict) -> dict | None:
    end = float(agent_turn.get("end") or 0.0)
    for turn in sorted(turns, key=lambda item: float(item.get("start") or 0.0)):
        if turn.get("channel") == 0 and float(turn.get("start") or 0.0) >= end:
            return turn
    return None


def extract_semantic_features(turns: list[dict]) -> dict[str, float]:
    """One row of probe features from ordered transcript turns."""
    agent = _channel_turns(turns, 1)
    caller = _channel_turns(turns, 0)
    traps = digit_trap_pairs(agent)

    objected = 0.0
    for original, mutated in traps:
        mutation_turn = next(
            (
                turn
                for turn in agent
                if mutated in _text(turn) and original not in _text(turn)
            ),
            None,
        )
        if mutation_turn is None:
            continue
        reply = caller_reply_after(turns, mutation_turn)
        if reply is not None and _OBJECTION.search(_text(reply)):
            objected += 1.0

    dilemma_count = sum(
        1.0
        for turn in agent
        if "?" in _text(turn) and _DILEMMA.search(_text(turn))
    )
    interrupt_count = sum(1.0 for turn in agent if _INTERRUPT.search(_text(turn)))
    caller_text = " ".join(_text(turn) for turn in caller)
    filler_count = float(len(_FILLERS.findall(caller_text)))

    return {
        "digit_trap_count": float(len(traps)),
        "digit_trap_caller_objected": objected,
        "false_dilemma_count": dilemma_count,
        "announced_interrupt_count": interrupt_count,
        "caller_filler_count": filler_count,
    }
