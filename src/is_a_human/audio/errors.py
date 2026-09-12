class AudioValidationError(ValueError):
    """Base error for invalid telephony audio input."""


class InvalidBase64Error(AudioValidationError):
    """Payload is not valid base64."""


class NotStereoError(AudioValidationError):
    """Audio stream is not dual-channel stereo."""


class SampleRateError(AudioValidationError):
    """Audio sample rate is not supported."""


class InvalidWavError(AudioValidationError):
    """Bytes are not a readable WAV container."""
