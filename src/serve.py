"""The deliverable: POST /detect.

Receives a base64 stereo WAV (8 kHz, ch0=caller, ch1=agent), runs the fusion of
whatever views are enabled, returns {is_synthetic, confidence, views}.

uvicorn src.serve:app --host 0.0.0.0 --port 8000
Test: curl -X POST localhost:8000/detect -H 'Content-Type: application/json' \
        -d '{"audio_b64":"<base64 wav>"}'
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.call import Call
from src.fusion import Fusion
from src.views.behavioral_view import BehavioralView
# from src.views.acoustic_view import AcousticView
# from src.views.semantic_view import SemanticView

app = FastAPI(title="is-a-human detector")

# must match the VIEWS list used in training
VIEWS = [BehavioralView()]
fusion = Fusion(VIEWS).load()


class DetectIn(BaseModel):
    audio_b64: str


class DetectOut(BaseModel):
    is_synthetic: bool
    confidence: float
    views: dict


@app.post("/detect", response_model=DetectOut)
def detect(inp: DetectIn):
    wav_bytes = base64.b64decode(inp.audio_b64)
    with tempfile.NamedTemporaryFile("wb", suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    call = Call.from_wav(tmp)
    p, per_view = fusion.proba(call)
    Path(tmp).unlink(missing_ok=True)
    return DetectOut(
        is_synthetic=bool(p >= 0.5),
        confidence=round(abs(p - 0.5) * 2, 4),   # 0=unsure, 1=certain
        views={k: round(v, 4) for k, v in per_view.items()},
    )


@app.get("/health")
def health():
    return {"ok": True, "views": [v.name for v in VIEWS]}
