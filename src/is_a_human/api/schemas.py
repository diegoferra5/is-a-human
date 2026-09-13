"""Request/response models for POST /detect (Altur judge contract)."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DetectRequest(BaseModel):
    """Judge body: audio_base64 plus optional metadata. audio_b64 is accepted too."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    audio_base64: str = Field(
        ...,
        validation_alias=AliasChoices("audio_base64", "audio_b64"),
        description="Base64-encoded stereo WAV, 8 kHz, ch0=caller, ch1=agent",
    )
    call_id: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class DetectResponse(BaseModel):
    """Judge-visible verdict. `views` holds per-head P(synthetic) for debugging only."""

    is_synthetic: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    views: dict[str, float] | None = None


def verdict_from_probability(p_synthetic: float) -> tuple[bool, float]:
    """Map P(synthetic) to the judge contract.

    `confidence` is certainty in `is_synthetic`, not P(synthetic). The judge
    recovers P(synthetic) as `confidence` when true and `1 - confidence` when false.
    """
    is_synthetic = bool(p_synthetic >= 0.5)
    confidence = float(p_synthetic if is_synthetic else 1.0 - p_synthetic)
    return is_synthetic, confidence
