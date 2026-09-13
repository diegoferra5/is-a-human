"""A Call object: the single unit every view receives.

Carries the audio + lazily-computed turns so each view can pull what it needs
(behavioral -> turns, acoustic -> waveform, semantic -> waveform for ASR).
"""

##this class gives each view what needs
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import AUDIO, CACHE
from src.vad import extract_turns, read_wav

_VAD_CACHE = CACHE / "vad_turns"
_VAD_CACHE.mkdir(exist_ok=True)


@dataclass
class Call:
    anon_id: str
    audio_path: Path
    duration_s: float | None = None
    _turns: dict | None = field(default=None, repr=False) ## these are the turns 
    _wav: tuple | None = field(default=None, repr=False)

    @classmethod
    def from_id(cls, anon_id: str, duration_s: float | None = None) -> "Call":
        return cls(anon_id, AUDIO / f"{anon_id}.wav", duration_s)

    @classmethod
    def from_wav(cls, path: str | Path) -> "Call":
        p = Path(path)
        return cls(p.stem, p)

    def turns(self) -> dict:
        """VAD-derived {channel,start,end} turns (cached to disk by id)."""
        if self._turns is not None:
            return self._turns
        cache_file = _VAD_CACHE / f"{self.anon_id}.json"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            self._turns = json.loads(cache_file.read_text())
        else:
            self._turns = extract_turns(self.audio_path)
            tmp = cache_file.with_suffix(f".tmp.{os.getpid()}")  # atomic write
            tmp.write_text(json.dumps(self._turns))
            os.replace(tmp, cache_file)
        return self._turns

    def audio(self) -> tuple[np.ndarray, int]:
        """(samples[n, channels] float32, sample_rate), lazily loaded."""
        if self._wav is None:
            self._wav = read_wav(self.audio_path)
        return self._wav

    def caller_audio(self) -> tuple[np.ndarray, int]:
        x, sr = self.audio()
        return x[:, 0], sr        # channel 0 = caller (the one we classify)

    def agent_audio(self) -> tuple[np.ndarray, int]:
        x, sr = self.audio()
        return x[:, 1], sr        # channel 1 = bank agent (reference)
