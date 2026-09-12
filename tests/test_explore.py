from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from is_a_human.analysis.explore import _ExploreProgress, run_exploratory_analysis
from is_a_human.analysis.features import CallFeatures, extract_call_features
from is_a_human.dataset.loader import TurnSegment


def _fake_features(label: str, caller_cv: float) -> CallFeatures:
    base = {name: 0.0 for name in CallFeatures.numeric_field_names()}
    base.update(
        {
            "caller_talk_ratio": 0.2,
            "agent_talk_ratio": 0.5,
            "caller_response_latency_cv": caller_cv,
        }
    )
    return CallFeatures(
        anon_id=f"call_{label}",
        label=label,
        split="val",
        **base,
    )


def _sample(label: str):
    return SimpleNamespace(
        anon_id=f"call_{label}",
        label=label,
        split="val",
        ch0_caller=np.array([], dtype=np.float32),
        ch1_agent=np.array([], dtype=np.float32),
        sample_rate=8000,
        turns=(TurnSegment(channel=1, start=0.0, end=1.0), TurnSegment(channel=0, start=2.0, end=3.0)),
    )


def test_explore_progress_writes_to_stderr(capsys):
    progress = _ExploreProgress("test", total=2, enabled=True)
    progress.update("call_a")
    progress.update("call_b")
    progress.done()
    assert "[explore] test: 2/2" in capsys.readouterr().err


def test_run_exploratory_analysis_ranks_features():
    samples = [
        _sample("human"),
        _sample("human"),
        _sample("synthetic"),
        _sample("synthetic"),
    ]
    feature_rows = [
        _fake_features("human", caller_cv=0.9),
        _fake_features("human", caller_cv=0.8),
        _fake_features("synthetic", caller_cv=0.4),
        _fake_features("synthetic", caller_cv=0.5),
    ]

    with patch("is_a_human.analysis.explore.iter_split", return_value=iter(samples)):
        with patch(
            "is_a_human.analysis.explore.extract_call_features",
            side_effect=feature_rows,
        ):
            summary = run_exploratory_analysis(
                split="val", dataset_root="resources/challenge-dataset"
            )

    assert summary.num_calls == 4
    assert summary.top_separators[0].feature == "caller_response_latency_cv"


@pytest.mark.integration
def test_extract_call_features_on_real_audio(real_call_sample):
    features = extract_call_features(
        anon_id=real_call_sample.anon_id,
        label=real_call_sample.label,
        split=real_call_sample.split,
        ch0_caller=real_call_sample.ch0_caller,
        ch1_agent=real_call_sample.ch1_agent,
        sample_rate=real_call_sample.sample_rate,
        organizer_turns=real_call_sample.turns,
    )

    assert features.duration_s > 0
    assert features.caller_rms_mean >= 0.0
    assert features.agent_aligned_recovery_count > 0
    assert features.caller_barge_in_count >= 0.0
