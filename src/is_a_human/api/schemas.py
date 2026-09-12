from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded stereo WAV, 8 kHz, ch0=caller, ch1=agent")


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
