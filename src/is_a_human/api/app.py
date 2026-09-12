"""FastAPI application for the /detect endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from is_a_human.api.schemas import DetectRequest, DetectResponse
from is_a_human.audio.demux import demux_base64_telephony
from is_a_human.audio.errors import AudioValidationError
from is_a_human.pipeline import process_call
from is_a_human.turns.vad import _get_vad_model


class AppState:
    ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_vad_model()
    state.ready = True
    yield
    state.ready = False


def create_app() -> FastAPI:
    app = FastAPI(title="is-a-human", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "ready": state.ready}

    @app.post("/detect", response_model=DetectResponse)
    async def detect(request: DetectRequest) -> DetectResponse:
        try:
            ch0, ch1, sample_rate = demux_base64_telephony(request.audio_b64)
            process_call(ch0, ch1, sample_rate)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Phase 0 stub: neutral placeholder until detection signals are added.
        return DetectResponse(is_synthetic=False, confidence=0.5)

    return app
