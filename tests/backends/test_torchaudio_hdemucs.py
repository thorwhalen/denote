"""Tests for the torchaudio Hybrid Demucs separation backend (thorwhalen/denote#7).

The model is replaced by a fake that returns known stems, so these need no
torchaudio and no weights; torch is needed for the tensor plumbing.
"""

import numpy as np
import pytest

import denote
from denote.base import SeparationResult
from denote.backends.torchaudio_hdemucs import adapter as hd

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

SR = hd.HDEMUCS_SR


class _FakeModel:
    """Returns each source as a fixed multiple of the (normalized) input."""

    sources = ["drums", "bass", "other", "vocals"]
    gains = [0.1, 0.2, 0.3, 0.4]

    def __init__(self):
        self.chunks = []

    def __call__(self, x):
        self.chunks.append(x.shape[-1])
        return torch.stack([g * x for g in self.gains], dim=1)


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(hd, "_model", lambda device: model)
    return model


def test_overlap_add_weights_sum_to_one():
    n, seg, ov = 1000, 300, 60
    total = np.zeros(n)
    start = 0
    while True:
        end = min(n, start + seg)
        total[start:end] += hd.overlap_add_weights(n, start, end, ov)
        if end == n:
            break
        start = end - ov
    assert np.allclose(total, 1.0)


def test_separate_chunks_and_reassembles_exactly(fake_model):
    rng = np.random.default_rng(0)
    mix = rng.normal(0, 0.1, (2, 3 * SR)).astype(np.float32)
    result = denote.separate(mix, sr=SR, segment_s=1.0, overlap_s=0.25)
    assert isinstance(result, SeparationResult) and result.sr == SR
    assert result.backend == "torchaudio_hdemucs"
    assert len(fake_model.chunks) == 4  # 3 s in 1 s chunks overlapping by 0.25 s
    # each stem is gain * (normalized mix), de-normalized: exactly reassembled
    mean, std = mix.mean(), mix.std() + 1e-8
    for name, g in zip(_FakeModel.sources, _FakeModel.gains):
        expected = g * (mix - mean) / std * std + mean
        assert np.allclose(result.stems[name], expected, atol=1e-5)
    assert np.allclose(result.vocals, result.stems["vocals"].mean(axis=0))


def test_separate_takes_mono_and_frames_first(fake_model):
    mono = np.sin(np.linspace(0, 100, SR)).astype(np.float32)
    assert denote.separate(mono, sr=SR).stems["vocals"].shape == (2, SR)
    frames_first = np.stack(
        [mono, mono], axis=1
    )  # (frames, channels), as soundfile reads
    assert denote.separate(frames_first, sr=SR).stems["bass"].shape == (2, SR)


def test_separate_resamples_to_the_model_rate(fake_model):
    mono = np.zeros(22050, dtype=np.float32)
    result = denote.separate(np.stack([mono, mono], axis=1), sr=22050)
    assert result.stems["drums"].shape == (2, SR)  # 1 s at 44.1 kHz, along time


def test_separate_rejects_a_segment_no_longer_than_its_overlap(fake_model):
    with pytest.raises(ValueError, match="overlap"):
        denote.separate(np.zeros(SR), sr=SR, segment_s=1.00001, overlap_s=1.0)


def test_separation_result_mono_names_its_stems():
    r = SeparationResult(stems={"vocals": np.ones((2, 4))}, sr=SR)
    with pytest.raises(KeyError, match="vocals"):
        r.mono("drums")


def test_separate_is_the_default_for_its_task():
    assert denote.get_default_backend("separate") == "torchaudio_hdemucs"
