from collections import Counter

import pytest

from is_a_human.dataset.errors import DatasetError
from is_a_human.dataset.loader import iter_split, load_call, load_manifest


def test_resolve_hackmty_layout(hackmty_paths):
    assert hackmty_paths.manifest_path.exists()
    assert hackmty_paths.turns_dir.is_dir()
    assert hackmty_paths.audio_dir.is_dir()


def test_load_manifest(hackmty_paths):
    rows = load_manifest(hackmty_paths.root)
    assert len(rows) == 353
    assert {row.split for row in rows} == {"train", "val"}
    assert {row.label for row in rows} == {"human", "synthetic"}


def test_manifest_split_counts(hackmty_paths):
    rows = load_manifest(hackmty_paths.root)
    splits = Counter(row.split for row in rows)
    labels = Counter(row.label for row in rows)

    assert splits["train"] == 282
    assert splits["val"] == 71
    assert labels["human"] == 150
    assert labels["synthetic"] == 203


def test_every_manifest_entry_has_turns_and_audio(hackmty_paths):
    rows = load_manifest(hackmty_paths.root)
    for row in rows:
        turns_path = hackmty_paths.turns_dir / f"{row.anon_id}.json"
        audio_path = hackmty_paths.audio_dir / f"{row.anon_id}.wav"
        assert turns_path.exists(), row.anon_id
        assert audio_path.exists(), row.anon_id


def test_load_call_with_turns(hackmty_paths):
    rows = load_manifest(hackmty_paths.root)
    sample = load_call(rows[0].anon_id, root=hackmty_paths.root, load_audio=False)
    assert sample.turns
    assert all(segment.channel in (0, 1) for segment in sample.turns)
    assert all(segment.end > segment.start for segment in sample.turns)


def test_load_call_with_audio(hackmty_paths):
    rows = load_manifest(hackmty_paths.root)
    sample = load_call(rows[0].anon_id, root=hackmty_paths.root, load_audio=True)
    assert sample.sample_rate == 8000
    assert sample.ch0_caller.shape == sample.ch1_agent.shape
    assert sample.ch0_caller.size > 0


def test_audio_duration_matches_manifest(hackmty_paths):
    sample = load_call("call_0181ce113ebe", root=hackmty_paths.root, load_audio=True)
    actual_duration_s = len(sample.ch0_caller) / sample.sample_rate
    assert abs(actual_duration_s - sample.duration_s) < 2.0


def test_iter_split_train_count(hackmty_paths):
    train_calls = list(iter_split("train", root=hackmty_paths.root, load_audio=False))
    assert len(train_calls) == 282
    assert all(call.split == "train" for call in train_calls)


def test_iter_split_val_count(hackmty_paths):
    val_calls = list(iter_split("val", root=hackmty_paths.root, load_audio=False))
    assert len(val_calls) == 71
    assert all(call.split == "val" for call in val_calls)


def test_load_call_unknown_id_raises(hackmty_paths):
    with pytest.raises(DatasetError, match="anon_id not in manifest"):
        load_call("call_does_not_exist", root=hackmty_paths.root)
