"""Build the behavioral feature table from a chosen turn source.

source="turns" -> provided turns.json (train-only sanity; not available at serve)
source="vad"   -> our VAD on the raw WAV (what we actually serve on)

VAD turns are cached under cache/vad_turns/ so we compute them once.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.config import MANIFEST, TURNS, AUDIO, CACHE
from src.features.behavioral import extract
from src.vad import extract_turns

VAD_CACHE = CACHE / "vad_turns"
VAD_CACHE.mkdir(exist_ok=True)


def turns_path_for(aid: str, source: str) -> Path:
    if source == "turns":
        return TURNS / f"{aid}.json"
    cache_file = VAD_CACHE / f"{aid}.json"
    # Treat empty/corrupt files (e.g. from an interrupted run) as missing.
    if not cache_file.exists() or cache_file.stat().st_size == 0:
        data = extract_turns(AUDIO / f"{aid}.wav")
        # Atomic write: fill a temp file, then rename it into place. A reader
        # never sees a half-written file, so parallel runs can't corrupt it.
        tmp = cache_file.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, cache_file)
    return cache_file


def build_table(source: str = "vad") -> pd.DataFrame:
    man = pd.read_csv(MANIFEST)
    rows = []
    for _, r in man.iterrows():
        feats = extract(turns_path_for(r.anon_id, source), duration_s=r.duration_s)
        feats["anon_id"] = r.anon_id
        feats["y"] = 1 if r.label == "synthetic" else 0
        feats["split"] = r.split
        rows.append(feats)
    return pd.DataFrame(rows)
