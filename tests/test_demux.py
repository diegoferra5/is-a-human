import base64

import numpy as np
import pytest

from is_a_human.audio.demux import demux_base64_telephony, demux_wav_bytes
from is_a_human.audio.errors import InvalidBase64Error, NotStereoError, SampleRateError


def test_demux_wav_bytes_splits_channels(stereo_wav_bytes: bytes):
    ch0, ch1, sample_rate = demux_wav_bytes(stereo_wav_bytes)

    assert sample_rate == 8000
    assert ch0.shape == ch1.shape
    assert ch0.dtype == np.float32
    assert ch1.dtype == np.float32
    assert not np.allclose(ch0, ch1)


def test_demux_base64_telephony(stereo_wav_b64: str):
    ch0, ch1, sample_rate = demux_base64_telephony(stereo_wav_b64)
    assert sample_rate == 8000
    assert ch0.shape == ch1.shape


def test_invalid_base64_raises():
    with pytest.raises(InvalidBase64Error):
        demux_base64_telephony("not-valid-base64!!!")


def test_unsupported_sample_rate_raises():
    import io
    import soundfile as sf

    stereo = np.column_stack([np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.float32)])
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 44100, format="WAV")
    with pytest.raises(SampleRateError):
        demux_wav_bytes(buffer.getvalue())


def test_mono_wav_raises_not_stereo():
    import io
    import soundfile as sf

    mono = np.zeros(8000, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, mono, 8000, format="WAV")
    with pytest.raises(NotStereoError):
        demux_wav_bytes(buffer.getvalue())
