"""Cross-channel interaction features: breath, entrainment, barge-in decay, echo."""

from __future__ import annotations

import numpy as np

from is_a_human.analysis.dsp import frame_rms, frame_signal, spectral_pink_slope
from is_a_human.dataset.loader import TurnSegment


def _frame_audio(audio: np.ndarray, start_s: float, end_s: float, sample_rate: int) -> np.ndarray:
    start_idx = max(0, int(start_s * sample_rate))
    end_idx = min(len(audio), int(end_s * sample_rate))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return audio[start_idx:end_idx]


def _segments_by_channel(turns: tuple[TurnSegment, ...], channel: int) -> list[tuple[float, float]]:
    return sorted(
        [(segment.start, segment.end) for segment in turns if segment.channel == channel],
        key=lambda item: item[0],
    )


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _non_speech_intervals(
    speech_segments: list[tuple[float, float]],
    duration_s: float,
) -> list[tuple[float, float]]:
    if not speech_segments:
        return [(0.0, duration_s)] if duration_s > 0 else []

    intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in speech_segments:
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_s:
        intervals.append((cursor, duration_s))
    return intervals


def _zero_crossing_rate(frame: np.ndarray) -> float:
    if frame.size < 2:
        return 0.0
    signs = np.sign(frame)
    return float(np.mean(signs[:-1] != signs[1:]))


def _gaps_before_long_utterances(
    caller_segments: list[tuple[float, float]],
    *,
    min_utterance_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Non-speech gaps immediately preceding caller segments longer than min_utterance_s."""
    gaps: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(caller_segments):
        if end - start < min_utterance_s:
            continue
        gap_start = caller_segments[index - 1][1] if index > 0 else 0.0
        if start - gap_start >= 0.03:
            gaps.append((gap_start, start))
    return gaps


def _is_breath_gap(
    ch0_caller: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> bool:
    """Pink-noise inhalation burst in a pre-utterance silence gap."""
    interval = _frame_audio(ch0_caller, start_s, end_s, sample_rate)
    min_samples = int(0.04 * sample_rate)
    if interval.size < min_samples:
        return False

    for frame in frame_signal(interval, sample_rate, frame_ms=30.0, hop_ms=15.0):
        duration = frame.size / sample_rate
        if duration < 0.04 or duration > 0.45:
            continue

        rms = frame_rms(frame)
        zcr = _zero_crossing_rate(frame)
        slope = spectral_pink_slope(frame, sample_rate)
        if 0.003 <= rms <= 0.06 and zcr >= 0.08 and slope < -0.5:
            return True

    return False


def _detect_breath_events(
    ch0_caller: np.ndarray,
    sample_rate: int,
    non_speech: list[tuple[float, float]],
) -> int:
    """Low-energy, pink-slope bursts in caller non-speech intervals."""
    breath_count = 0
    for start_s, end_s in non_speech:
        if _is_breath_gap(ch0_caller, sample_rate, start_s, end_s):
            breath_count += 1
    return breath_count


def _energy_envelope(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    values = [frame_rms(frame) for frame in frame_signal(audio, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)]
    return np.asarray(values, dtype=np.float64)


def _cross_channel_energy_correlation(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
) -> float:
    env0 = _energy_envelope(ch0_caller, sample_rate)
    env1 = _energy_envelope(ch1_agent, sample_rate)
    length = min(env0.size, env1.size)
    if length < 3:
        return 0.0

    a = env0[:length]
    b = env1[:length]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def _agent_interrupts_caller(
    caller_segments: list[tuple[float, float]],
    agent_segments: list[tuple[float, float]],
) -> list[float]:
    """Return agent start times where agent barges into ongoing caller speech."""
    triggers: list[float] = []
    for agent_start, agent_end in agent_segments:
        for caller_start, caller_end in caller_segments:
            if caller_start < agent_start < caller_end:
                triggers.append(agent_start)
                break
    return triggers


def _yield_decay_ms(
    ch0_caller: np.ndarray,
    sample_rate: int,
    interrupt_time_s: float,
    *,
    window_ms: float = 150.0,
) -> float:
    """Caller RMS decay (dB) over first 150 ms after agent interruption."""
    start_idx = int(interrupt_time_s * sample_rate)
    window_samples = int(sample_rate * window_ms / 1000.0)
    end_idx = min(len(ch0_caller), start_idx + window_samples)
    if end_idx - start_idx < 8:
        return 0.0

    segment = ch0_caller[start_idx:end_idx].astype(np.float64)
    quarter = max(1, segment.size // 4)
    early = frame_rms(segment[:quarter])
    late = frame_rms(segment[-quarter:])
    if early <= 1e-12:
        return 0.0

    decay_db = 20.0 * np.log10(max(late, 1e-12) / early)
    return float(decay_db)


def _agent_only_intervals(
    caller_segments: list[tuple[float, float]],
    agent_segments: list[tuple[float, float]],
    duration_s: float,
) -> list[tuple[float, float]]:
    """Intervals where the agent speaks and the caller does not."""
    if not agent_segments:
        return []

    intervals: list[tuple[float, float]] = []
    for agent_start, agent_end in agent_segments:
        cursor = agent_start
        while cursor < agent_end:
            next_caller_start = agent_end
            for caller_start, caller_end in caller_segments:
                if caller_end <= cursor or caller_start >= agent_end:
                    continue
                if caller_start > cursor:
                    next_caller_start = min(next_caller_start, caller_start)
                if caller_start <= cursor < caller_end:
                    cursor = min(caller_end, agent_end)
                    next_caller_start = agent_end
                    break
            else:
                if next_caller_start > cursor:
                    intervals.append((cursor, min(next_caller_start, agent_end)))
                cursor = next_caller_start
    return intervals


def _segment_bleed_correlation(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    *,
    min_lag_ms: float = 5.0,
    max_lag_ms: float = 200.0,
) -> float:
    """Agent→caller bleed: cross-correlation at positive lags within one interval."""
    caller = _frame_audio(ch0_caller, start_s, end_s, sample_rate).astype(np.float64)
    agent = _frame_audio(ch1_agent, start_s, end_s, sample_rate).astype(np.float64)
    if caller.size < 64 or agent.size < 64:
        return 0.0

    caller -= np.mean(caller)
    agent -= np.mean(agent)
    min_lag = max(1, int(sample_rate * min_lag_ms / 1000.0))
    max_lag = min(caller.size // 4, int(sample_rate * max_lag_ms / 1000.0))
    if max_lag <= min_lag:
        return 0.0

    denom = float(np.linalg.norm(caller) * np.linalg.norm(agent))
    if denom <= 1e-12:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        corr = float(np.dot(caller[lag:], agent[:-lag])) / denom
        best = max(best, abs(corr))
    return best


def _agent_echo_correlation(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    *,
    min_lag_ms: float = 5.0,
    max_lag_ms: float = 200.0,
) -> float:
    """Max normalized cross-correlation at positive lags (whole-call room bleed)."""
    if ch0_caller.size < 64 or ch1_agent.size < 64:
        return 0.0

    length = min(ch0_caller.size, ch1_agent.size)
    return _segment_bleed_correlation(
        ch0_caller,
        ch1_agent,
        sample_rate,
        0.0,
        length / sample_rate,
        min_lag_ms=min_lag_ms,
        max_lag_ms=max_lag_ms,
    )


def _agent_only_bleed_correlation(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    agent_only: list[tuple[float, float]],
) -> float:
    """Mean peak lagged correlation during agent-only speech (physical echo)."""
    correlations = [
        _segment_bleed_correlation(ch0_caller, ch1_agent, sample_rate, start, end)
        for start, end in agent_only
        if end - start >= 0.08
    ]
    if not correlations:
        return 0.0
    return float(np.mean(correlations))


def extract_interaction_physics_features(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    organizer_turns: tuple[TurnSegment, ...],
    duration_s: float,
) -> dict[str, float]:
    """Breath rate, entrainment, barge-in decay, acoustic echo leakage."""
    caller_segments = _segments_by_channel(organizer_turns, channel=0)
    agent_segments = _segments_by_channel(organizer_turns, channel=1)

    caller_speech_time = sum(end - start for start, end in caller_segments)
    non_speech = _non_speech_intervals(caller_segments, duration_s)
    pre_long_gaps = _gaps_before_long_utterances(caller_segments)
    breath_count = _detect_breath_events(ch0_caller, sample_rate, non_speech)
    breath_rate = breath_count / caller_speech_time if caller_speech_time > 0 else 0.0

    pre_long_breath_count = sum(
        1 for gap in pre_long_gaps if _is_breath_gap(ch0_caller, sample_rate, gap[0], gap[1])
    )
    breath_gap_ratio = (
        pre_long_breath_count / len(pre_long_gaps) if pre_long_gaps else 0.0
    )

    agent_only = _agent_only_intervals(caller_segments, agent_segments, duration_s)
    agent_bleed = _agent_only_bleed_correlation(ch0_caller, ch1_agent, sample_rate, agent_only)

    entrainment = _cross_channel_energy_correlation(ch0_caller, ch1_agent, sample_rate)

    decay_values = [
        _yield_decay_ms(ch0_caller, sample_rate, trigger)
        for trigger in _agent_interrupts_caller(caller_segments, agent_segments)
    ]
    decay_mean = float(np.mean(decay_values)) if decay_values else 0.0
    decay_std = float(np.std(decay_values)) if decay_values else 0.0

    echo_corr = _agent_echo_correlation(ch0_caller, ch1_agent, sample_rate)

    return {
        "caller_breath_event_rate": float(breath_rate),
        "caller_breath_gap_ratio": float(breath_gap_ratio),
        "cross_channel_energy_correlation": entrainment,
        "caller_yield_decay_mean_db": decay_mean,
        "caller_yield_decay_std_db": decay_std,
        "caller_agent_echo_correlation": echo_corr,
        "caller_agent_bleed_correlation": agent_bleed,
    }
