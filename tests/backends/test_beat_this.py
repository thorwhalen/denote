"""Tests for the Beat This! beat and downbeat backend (thorwhalen/denote#13).

The plumbing tests replace the model with a fake, so they need neither beat_this
nor torch and run everywhere. The model test runs only where beat_this is installed
(it downloads the 'final0' checkpoint on first use).
"""

import numpy as np
import pytest

import denote
from denote.base import BeatResult
from denote.backends.beat_this import adapter as bt

from .synthetic import click_track, f_measure


class _FakeTracker:
    """Records what it was given; returns fixed beats and downbeats."""

    def __init__(self, beats, downbeats):
        self.beats, self.downbeats, self.calls = beats, downbeats, []

    def __call__(self, signal, sr):
        self.calls.append((signal, sr))
        return self.beats, self.downbeats


@pytest.fixture
def fake_tracker(monkeypatch):
    tracker = _FakeTracker(np.arange(0.5, 4.0, 0.5), np.arange(0.5, 4.0, 2.0))
    loads = []

    def load(*args):
        loads.append(args)
        return tracker

    monkeypatch.setattr(bt, "_load", load)
    tracker.loads = loads
    return tracker


def test_get_beats_returns_beats_downbeats_and_tempo(fake_tracker):
    y = np.zeros(44100, dtype=np.float32)
    result = denote.get_beats(y, sr=44100, backend="beat_this")
    assert isinstance(result, BeatResult) and result.backend == "beat_this"
    assert np.allclose(result.beats, fake_tracker.beats)
    assert np.allclose(result.downbeats, fake_tracker.downbeats)
    assert result.tempo == pytest.approx(120.0)  # median inter-beat interval 0.5 s


def test_audio_reaches_the_model_mono_at_its_rate(fake_tracker):
    stereo = np.zeros((2, 44100), dtype=np.float32)  # 1 s at 44.1 kHz
    denote.get_beats(stereo, sr=44100, backend="beat_this")
    signal, sr = fake_tracker.calls[-1]
    assert sr == bt.BEAT_THIS_SR and signal.shape == (bt.BEAT_THIS_SR,)
    assert signal.dtype == np.float32


def test_options_translate_to_the_checkpoint_and_device(fake_tracker):
    y = np.zeros(22050, dtype=np.float32)
    denote.get_beats(y, sr=22050, backend="beat_this")
    denote.get_beats(y, sr=22050, backend="beat_this", model="small0", dbn=True)
    assert fake_tracker.loads == [
        ("final0", "cpu", False, False),
        ("small0", "cpu", True, False),
    ]


def test_tempo_uses_the_median_interval_so_an_outlier_gap_does_not_drag_it(
    monkeypatch,
):
    # 0.5 s intervals with jitter, then one 1.5 s gap (a missed beat): the mean
    # interval (0.7 s) would read 86 BPM; the median reads 120.
    beats = np.array([0.0, 0.49, 0.99, 1.5, 2.0, 3.5])
    monkeypatch.setattr(bt, "_load", lambda *a: _FakeTracker(beats, beats[:1]))
    result = denote.get_beats(
        np.zeros(22050, np.float32), sr=22050, backend="beat_this"
    )
    assert result.tempo == pytest.approx(120.0)
    assert 60 / np.mean(np.diff(beats)) == pytest.approx(85.7, abs=0.1)


def test_tempo_is_none_with_fewer_than_two_beats():
    assert bt.tempo_from_beats(np.array([1.0])) is None
    assert bt.tempo_from_beats(np.array([0.0, 0.25, 0.5])) == pytest.approx(240.0)


def test_beat_this_finds_the_beats_and_downbeats_of_a_click_track():
    pytest.importorskip("beat_this")
    y, sr, beats, downbeats = click_track(bpm=120.0, seconds=8.0)
    result = denote.get_beats(y, sr=sr, backend="beat_this")
    assert f_measure(result.beats, beats) > 0.9
    assert result.tempo == pytest.approx(120.0, rel=0.03)
    assert len(result.downbeats) > 0
    assert f_measure(result.downbeats, downbeats) > 0.5
