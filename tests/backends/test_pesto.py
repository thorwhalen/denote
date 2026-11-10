"""Tests for the PESTO pitch backend (thorwhalen/denote#13).

The plumbing tests replace the model (``_load``) and its torch call (``_predict``)
with fakes, so they need neither pesto nor torch and run everywhere. The model
tests run only where pesto-pitch is installed (its checkpoints ship in the wheel:
no download).
"""

import numpy as np
import pytest

import denote
from denote.base import PitchResult
from denote.backends.pesto import adapter as pa
from denote.backends.pesto.adapter import frame_period

from .synthetic import away_from_changes, cents_error, harmonic_tone


def test_frame_period_uses_the_rounded_hop():
    assert frame_period(10.0, 16000) == 0.01
    assert frame_period(10.0, 22050) == pytest.approx(221 / 22050)


@pytest.fixture
def fake_pesto(monkeypatch):
    calls = {"load": [], "predict": []}

    def load(*key):
        calls["load"].append(key)
        return "model"

    def predict(model, y, sr, device):
        calls["predict"].append((model, y, sr, device))
        n = 5
        return {
            "pitch": np.full(n, 330.0),
            "confidence": np.array([0.9, 0.1, 0.6, 0.5, 0.49]),
            "volume": np.ones(n),
            "activations": np.zeros((n, 3)),
        }

    monkeypatch.setattr(pa, "_load", load)
    monkeypatch.setattr(pa, "_predict", predict)
    return calls


def test_adapter_times_threshold_and_cache_key(fake_pesto):
    quiet = 0.03 * np.sin(np.linspace(0, 300, 22050)).astype(np.float32)
    result = denote.get_pitch(quiet, sr=22050, backend="pesto", reduction="argmax")
    assert isinstance(result, PitchResult) and result.backend == "pesto"
    assert result.times[1] == pytest.approx(221 / 22050)  # 10 ms rounded to 221 samples
    assert np.array_equal(
        np.isnan(result.frequencies), [False, True, False, False, True]
    )
    # everything the model holds as state is in its cache key
    assert fake_pesto["load"][-1] == ("mir-1k_g7", 10.0, 22050, "argmax", "cpu")
    _, y, sr, device = fake_pesto["predict"][-1]
    assert sr == 22050 and device == "cpu"
    assert np.max(np.abs(y)) == pytest.approx(1.0)  # peak-normalised by default


def test_normalize_false_passes_the_level_through(fake_pesto):
    quiet = 0.03 * np.sin(np.linspace(0, 300, 22050)).astype(np.float32)
    denote.get_pitch(quiet, sr=22050, backend="pesto", normalize=False)
    y = fake_pesto["predict"][-1][1]
    assert np.max(np.abs(y)) == pytest.approx(0.03, rel=1e-3)


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


def test_pesto_voices_a_quiet_tone_at_minus_30_dbfs():
    pytest.importorskip("pesto")
    y, sr, f0_at = harmonic_tone()
    y = y * (10 ** (-30 / 20) / np.max(np.abs(y)))
    result = denote.get_pitch(y, sr=sr, backend="pesto")
    keep = away_from_changes(result.times, end=len(y) / sr)
    voiced = keep & ~np.isnan(result.frequencies)
    assert voiced.sum() > 0.9 * keep.sum()
    err = cents_error(result.frequencies[voiced], f0_at(result.times[voiced]))
    assert np.median(err) < 20


def test_pesto_times_use_the_rounded_hop():
    pytest.importorskip("pesto")
    y, sr, _ = harmonic_tone()
    result = denote.get_pitch(y, sr=sr, backend="pesto", step_size_ms=10.0)
    assert result.times[1] == pytest.approx(221 / 22050)
    assert abs(result.times[-1] - len(y) / sr) < 0.05


def test_pesto_calls_noise_unvoiced():
    pytest.importorskip("pesto")
    noise = 0.1 * np.random.default_rng(0).normal(size=22050).astype(np.float32)
    result = denote.get_pitch(noise, sr=22050, backend="pesto")
    assert np.mean(np.isnan(result.frequencies)) > 0.9
