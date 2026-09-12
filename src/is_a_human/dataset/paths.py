"""Resolve manifest, turns, and audio paths for local development layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from is_a_human.dataset.errors import DatasetError


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    manifest_path: Path
    turns_dir: Path
    audio_dir: Path


def _repo_root() -> Path:
    return Path.cwd()


def _candidate_roots(explicit_root: Path | None) -> list[Path]:
    if explicit_root is not None:
        return [explicit_root]

    repo = _repo_root()
    return [
        repo / "dataset",
        repo / "resources" / "challenge-dataset",
        repo / "resources" / "hackmty26-main",
    ]


def _candidate_audio_dirs(root: Path, repo: Path) -> list[Path]:
    return [
        root / "audio",
        repo / "audio",
        repo / "resources" / "audio",
        repo / "dataset" / "audio",
    ]


def resolve_dataset_paths(root: Path | str | None = None) -> DatasetPaths:
    """Find manifest, turns, and audio directories across common repo layouts."""
    explicit = Path(root) if root is not None else None
    repo = _repo_root()

    for candidate in _candidate_roots(explicit):
        manifest_path = candidate / "manifest.csv"
        turns_dir = candidate / "turns"
        if not manifest_path.exists() or not turns_dir.is_dir():
            continue

        for audio_dir in _candidate_audio_dirs(candidate, repo):
            if audio_dir.is_dir():
                return DatasetPaths(
                    root=candidate,
                    manifest_path=manifest_path,
                    turns_dir=turns_dir,
                    audio_dir=audio_dir,
                )

        raise DatasetError(
            f"Found manifest and turns under {candidate}, but no audio directory. "
            "Expected audio/ with WAV files."
        )

    searched = ", ".join(str(path) for path in _candidate_roots(explicit))
    raise DatasetError(f"Dataset not found. Searched: {searched}")
