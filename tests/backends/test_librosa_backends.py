"""Integration tests for librosa-based backends (pyin, beats).

These always run since librosa is a core dependency.
"""

import pytest
import numpy as np


@pytest.fixture
def sine_array():
    """A 1-second 440Hz sine wave as a numpy array."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    return audio, sr


@pytest.fixture
def sine_wav(tmp_path):
    """Create a short sine wave WAV file."""
    import soundfile as sf

    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    # A melody-like signal: 440 Hz
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    path = tmp_path / 'sine.wav'
    sf.write(str(path), audio, sr)
    return str(path)


class TestLibrosaPyin:
    def test_get_pitch_from_array(self, sine_array):
        import denote
        from denote.base import PitchResult

        audio, sr = sine_array
        result = denote.get_pitch(audio, sr=sr, backend='librosa_pyin')
        assert isinstance(result, PitchResult)
        assert result.backend == 'librosa_pyin'
        assert len(result.times) > 0
        assert len(result.frequencies) == len(result.times)
        # The dominant frequency should be near 440 Hz
        voiced = ~np.isnan(result.frequencies)
        if voiced.any():
            median_f0 = np.nanmedian(result.frequencies)
            assert 400 < median_f0 < 480

    def test_get_pitch_from_file(self, sine_wav):
        import denote

        result = denote.get_pitch(sine_wav, backend='librosa_pyin')
        assert len(result.times) > 0


class TestLibrosaBeats:
    def test_get_beats_from_file(self, sine_wav):
        import denote
        from denote.base import BeatResult

        result = denote.get_beats(sine_wav, backend='librosa_beats')
        assert isinstance(result, BeatResult)
        assert result.backend == 'librosa_beats'
        assert result.tempo is not None
        # A pure sine has no rhythmic content, so tempo may be 0
        assert result.tempo >= 0

    def test_get_beats_from_array(self, sine_array):
        import denote

        audio, sr = sine_array
        result = denote.get_beats(audio, sr=sr, backend='librosa_beats')
        assert result.tempo is not None
