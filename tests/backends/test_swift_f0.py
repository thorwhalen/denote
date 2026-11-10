"""Tests for the SwiftF0 pitch backend (thorwhalen/denote#13).

The plumbing tests replace the detector with a fake, so they need no swift-f0 and
run everywhere. The model tests run only where swift-f0 is installed (its ONNX model
ships in the wheel: no download).
"""

from types import SimpleNamespace

import numpy as np
import pytest

import denote
from denote.base import PitchResult
from denote.backends.swift_f0 import adapter as sf

from .synthetic import away_from_changes, cents_error, harmonic_tone


class _FakeDetector:
    def __init__(self):
        self.calls = []

    def detect(self, audio, sample_rate, fmin=None, fmax=None):
        self.calls.append((audio, sample_rate, fmin, fmax))
        n = 4
        return SimpleNamespace(
            timestamps=np.arange(n) * 0.016,
            pitch_hz=np.full(n, 220.0),
            confidence=np.array([0.9, 0.2, 0.5, 0.49]),
            loudness_db=np.zeros(n),
        )


@pytest.fixture
def fake_detector(monkeypatch):
    detector = _FakeDetector()
    monkeypatch.setattr(sf, "_load", lambda threads: detector)
    return detector


def test_low_confidence_frames_read_nan(fake_detector):
    result = denote.get_pitch(np.zeros(16000), sr=16000, backend="swift_f0")
    assert isinstance(result, PitchResult) and result.backend == "swift_f0"
    assert np.array_equal(np.isnan(result.frequencies), [False, True, False, True])
    assert np.allclose(result.confidence, [0.9, 0.2, 0.5, 0.49])
    strict = denote.get_pitch(
        np.zeros(16000), sr=16000, backend="swift_f0", confidence_threshold=0.95
    )
    assert np.isnan(strict.frequencies).all()


def test_audio_reaches_the_model_mono_at_16k_with_the_frequency_range(fake_detector):
    stereo = np.zeros((44100, 2), dtype=np.float32)  # 1 s, frames first
    denote.get_pitch(
        stereo, sr=44100, backend="swift_f0", min_frequency=80, max_frequency=900
    )
    audio, rate, fmin, fmax = fake_detector.calls[-1]
    assert rate == sf.SWIFT_F0_SR and audio.shape == (sf.SWIFT_F0_SR,)
    assert (fmin, fmax) == (80, 900)


def test_swift_f0_tracks_known_pitches():
    pytest.importorskip("swift_f0")
    y, sr, f0_at = harmonic_tone()
    result = denote.get_pitch(y, sr=sr, backend="swift_f0")
    keep = away_from_changes(result.times, end=len(y) / sr)
    voiced = keep & ~np.isnan(result.frequencies)
    assert voiced.sum() > 0.9 * keep.sum()
    err = cents_error(result.frequencies[voiced], f0_at(result.times[voiced]))
    assert np.median(err) < 20 and np.percentile(err, 95) < 50


def test_swift_f0_calls_silence_unvoiced():
    pytest.importorskip("swift_f0")
    result = denote.get_pitch(np.zeros(16000), sr=16000, backend="swift_f0")
    assert np.isnan(result.frequencies).all()
