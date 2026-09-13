"""FastAPI application for the /detect endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from is_a_human.analysis.features import extract_call_features
from is_a_human.api.schemas import DetectRequest, DetectResponse, verdict_from_probability
from is_a_human.audio.demux import demux_base64_telephony
from is_a_human.audio.errors import AudioValidationError
from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, TandemModel, load_tandem
from is_a_human.pipeline import process_call
from is_a_human.turns.vad import _get_vad_model


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
        _get_vad_model()
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
            result = process_call(ch0, ch1, sample_rate)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Tandem model not loaded. Run is-a-human-train.",
            )

        features = extract_call_features(
            anon_id=request.call_id or "live",
            label="unknown",
            split="live",
            ch0_caller=ch0,
            ch1_agent=ch1,
            sample_rate=sample_rate,
            pipeline_result=result,
        )
        probability, views = state.model.predict(features)
        is_synthetic, confidence = verdict_from_probability(probability)
        return DetectResponse(
            is_synthetic=is_synthetic,
            confidence=confidence,
            views=views,
        )

    return app
