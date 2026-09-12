"""Challenge dataset loader for local train/val iteration."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from is_a_human.audio.demux import load_stereo_wav
from is_a_human.dataset.errors import DatasetError
from is_a_human.dataset.paths import DatasetPaths, resolve_dataset_paths

Split = Literal["train", "val"]
Label = Literal["human", "synthetic"]


@dataclass(frozen=True)
class TurnSegment:
    channel: int
    start: float
    end: float


@dataclass(frozen=True)
class CallSample:
    anon_id: str
    label: Label
    split: Split
    duration_s: float
    ch0_caller: np.ndarray
    ch1_agent: np.ndarray
    sample_rate: int
    turns: tuple[TurnSegment, ...]


@dataclass(frozen=True)
class ManifestRow:
    anon_id: str
    label: Label
    split: Split
    duration_s: float


def load_manifest(root: Path | str | None = None) -> list[ManifestRow]:
    """Load manifest.csv rows."""
    paths = resolve_dataset_paths(root)
    rows: list[ManifestRow] = []
    with paths.manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ManifestRow(
                    anon_id=row["anon_id"],
                    label=row["label"],  # type: ignore[arg-type]
                    split=row["split"],  # type: ignore[arg-type]
                    duration_s=float(row["duration_s"]),
                )
            )
    return rows


def _load_turns(paths: DatasetPaths, anon_id: str) -> tuple[TurnSegment, ...]:
    turns_path = paths.turns_dir / f"{anon_id}.json"
    if not turns_path.exists():
        raise DatasetError(f"Turns file not found: {turns_path}")

    payload = json.loads(turns_path.read_text(encoding="utf-8"))
    segments = []
    for turn in payload.get("turns", []):
        segments.append(
            TurnSegment(
                channel=int(turn["channel"]),
                start=float(turn["start"]),
                end=float(turn["end"]),
            )
        )
    return tuple(segments)


def load_call(
    anon_id: str,
    root: Path | str | None = None,
    *,
    load_audio: bool = True,
) -> CallSample:
    """Load a single call by anon_id."""
    paths = resolve_dataset_paths(root)
    rows = {row.anon_id: row for row in load_manifest(root)}
    if anon_id not in rows:
        raise DatasetError(f"anon_id not in manifest: {anon_id}")

    row = rows[anon_id]
    audio_path = paths.audio_dir / f"{anon_id}.wav"
    if load_audio and not audio_path.exists():
        raise DatasetError(f"Audio file not found: {audio_path}")

    if load_audio:
        ch0, ch1, sample_rate = load_stereo_wav(str(audio_path))
    else:
        ch0 = np.array([], dtype=np.float32)
        ch1 = np.array([], dtype=np.float32)
        sample_rate = 8000

    turns = _load_turns(paths, anon_id)
    return CallSample(
        anon_id=row.anon_id,
        label=row.label,
        split=row.split,
        duration_s=row.duration_s,
        ch0_caller=ch0,
        ch1_agent=ch1,
        sample_rate=sample_rate,
        turns=turns,
    )


def iter_split(
    split: Split,
    root: Path | str | None = None,
    *,
    load_audio: bool = True,
) -> Iterator[CallSample]:
    """Iterate calls for a given split."""
    for row in load_manifest(root):
        if row.split != split:
            continue
        yield load_call(row.anon_id, root=root, load_audio=load_audio)
