"""FastAPI application for the /detect endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException

from is_a_human.api.schemas import DetectRequest, DetectResponse, verdict_from_probability
from is_a_human.audio.demux import demux_base64_telephony
from is_a_human.audio.errors import AudioValidationError
from is_a_human.detect.live import detect_from_audio, log_layer_timings
from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, TandemModel, load_tandem


class AppState:
    ready: bool = False
    model: TandemModel | None = None


state = AppState()


def _load_model(model_path: Path) -> TandemModel | None:
    if not model_path.exists():
        return None
    return load_tandem(model_path)


def create_app(model_path: Path | str | None = None) -> FastAPI:
    resolved = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    state.model = _load_model(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.model is None:
            state.model = _load_model(resolved)
        state.ready = True
        yield
        state.ready = False
        state.model = None

    app = FastAPI(title="is-a-human", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "ready": state.ready,
            "model_loaded": state.model is not None,
        }

    @app.post("/detect", response_model=DetectResponse)
    async def detect(request: DetectRequest) -> DetectResponse:
        try:
            ch0, ch1, sample_rate = demux_base64_telephony(request.audio_base64)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Tandem model not loaded. Run is-a-human-train.",
            )

        started = perf_counter()
        call_id = request.call_id or "live"
        probability, views, timings = detect_from_audio(
            state.model, ch0, ch1, sample_rate, call_id=call_id, heavy=False
        )
        log_layer_timings(call_id, timings, total_ms=(perf_counter() - started) * 1000)
        is_synthetic, confidence = verdict_from_probability(probability)
        return DetectResponse(
            is_synthetic=is_synthetic,
            confidence=confidence,
            views=views,
        )

    return app
