from pathlib import Path

import pytest

from is_a_human.dataset.errors import DatasetError
from is_a_human.dataset.paths import resolve_dataset_paths


def test_auto_resolve_from_repo_root(dataset_paths):
    resolved = resolve_dataset_paths()
    assert resolved.root.resolve() == dataset_paths.root.resolve()
    assert resolved.audio_dir.resolve() == dataset_paths.audio_dir.resolve()


def test_resolve_missing_dataset_raises():
    with pytest.raises(DatasetError, match="Dataset not found"):
        resolve_dataset_paths(Path("/nonexistent/dataset/root"))
