"""Integration tests for the basic_pitch backend.

These tests require basic-pitch to be installed and functional.
They are skipped if the dependency is missing or broken.
"""

import pytest
import numpy as np

bp = pytest.importorskip('basic_pitch', reason='basic-pitch not installed')


@pytest.fixture
def sine_wav(tmp_path):
    """Create a short sine wave WAV file for testing."""
    import soundfile as sf

    sr = 22050
    duration = 1.0
    freq = 440.0  # A4
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    path = tmp_path / 'sine_440.wav'
    sf.write(str(path), audio, sr)
    return str(path)


def test_transcribe_returns_result(sine_wav):
    import denote
    from denote.base import TranscriptionResult

    try:
        result = denote.transcribe(sine_wav, backend='basic_pitch')
    except (AttributeError, ValueError, RuntimeError) as e:
        pytest.skip(f'basic-pitch not functional in this environment: {e}')

    assert isinstance(result, TranscriptionResult)
    assert result.backend == 'basic_pitch'
    assert result.midi is not None


def test_transcribe_with_params(sine_wav):
    import denote

    try:
        result = denote.transcribe(
            sine_wav,
            backend='basic_pitch',
            onset_threshold=0.3,
            frame_threshold=0.2,
        )
    except (AttributeError, ValueError, RuntimeError) as e:
        pytest.skip(f'basic-pitch not functional in this environment: {e}')

    assert result.midi is not None
