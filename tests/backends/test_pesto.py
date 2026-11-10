"""Tests for the PESTO pitch backend (thorwhalen/denote#13).

PESTO needs torch, so all of these run only where pesto-pitch is installed (its
checkpoints ship in the wheel: no download).
"""

import numpy as np
import pytest

import denote
from denote.base import PitchResult
from denote.backends.pesto.adapter import frame_period

from .synthetic import away_from_changes, cents_error, harmonic_tone


def test_frame_period_uses_the_rounded_hop():
    assert frame_period(10.0, 16000) == 0.01
    assert frame_period(10.0, 22050) == pytest.approx(221 / 22050)


def test_pesto_tracks_known_pitches():
    pytest.importorskip("pesto")
    y, sr, f0_at = harmonic_tone()
    result = denote.get_pitch(y, sr=sr, backend="pesto")
    assert isinstance(result, PitchResult) and result.backend == "pesto"
    assert len(result.times) == len(result.frequencies) == len(result.confidence)
    keep = away_from_changes(result.times, end=len(y) / sr)
    voiced = keep & ~np.isnan(result.frequencies)
    assert voiced.sum() > 0.9 * keep.sum()
    err = cents_error(result.frequencies[voiced], f0_at(result.times[voiced]))
    assert np.median(err) < 20 and np.percentile(err, 95) < 50


def test_pesto_times_follow_the_audio_duration():
    pytest.importorskip("pesto")
    y, sr, _ = harmonic_tone()
    result = denote.get_pitch(y, sr=sr, backend="pesto", step_size_ms=20.0)
    assert result.times[1] == pytest.approx(frame_period(20.0, sr))
    assert abs(result.times[-1] - len(y) / sr) < 0.05


def test_pesto_calls_noise_unvoiced():
    pytest.importorskip("pesto")
    noise = 0.1 * np.random.default_rng(0).normal(size=22050).astype(np.float32)
    result = denote.get_pitch(noise, sr=22050, backend="pesto")
    assert np.mean(np.isnan(result.frequencies)) > 0.9
