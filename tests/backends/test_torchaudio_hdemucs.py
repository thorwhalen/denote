"""Tests for the torchaudio Hybrid Demucs separation backend (thorwhalen/denote#7).

The model is replaced by a fake that returns known stems and enforces the real
model's input constraints, so these need no torchaudio and no weights; torch is
needed for the tensor plumbing.
"""

import tracemalloc
import warnings

import numpy as np
import pytest

import denote
from denote.base import SeparationResult
from denote.backends.torchaudio_hdemucs import adapter as hd

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

SR = 44100  # both HDemucs bundles' rate; the fake reports it as the bundle would


def _real_model_accepts(frames: int) -> bool:
    # Measured on HDEMUCS_HIGH_MUSDB_PLUS (torchaudio 2.9): other lengths raise in _spec.
    return 769 <= frames <= 1024 or frames >= 1281


class _FakeModel:
    """Returns each source as a fixed multiple of the (normalized) input."""

    sources = ["drums", "bass", "other", "vocals"]
    gains = [0.1, 0.2, 0.3, 0.4]  # sum to one: the stems reassemble the input

    def __init__(self):
        self.chunks = []

    def __call__(self, x):
        assert x.ndim == 3 and x.shape[1] == 2, x.shape
        if not _real_model_accepts(x.shape[-1]):
            raise ValueError("Spectrogram's last dimension must be 4 + input size ...")
        self.chunks.append(x.shape[-1])
        return torch.stack([g * x for g in self.gains], dim=1)


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(hd, "_load", lambda bundle, device: (model, SR))
    return model


def _expected(mix, gain):
    mean, std = mix.mean(), mix.std() + hd.NORM_EPS
    return gain * (mix - mean) / std * std + mean / len(_FakeModel.sources)


def test_separate_chunks_and_reassembles_exactly(fake_model):
    rng = np.random.default_rng(0)
    mix = rng.normal(0.05, 0.1, (2, 3 * SR)).astype(np.float32)  # with a DC offset
    result = denote.separate(mix, sr=SR, segment_s=1.0, overlap_s=0.25)
    assert isinstance(result, SeparationResult) and result.sr == SR
    assert result.backend == "torchaudio_hdemucs"
    assert fake_model.chunks == [SR] * 4  # 3 s in 1 s chunks overlapping >= 0.25 s
    for name, g in zip(_FakeModel.sources, _FakeModel.gains):
        assert np.allclose(result.stems[name], _expected(mix, g), atol=1e-5)
    assert np.allclose(result.vocals, result.stems["vocals"].mean(axis=0))


def test_stems_sum_to_the_mix(fake_model):
    mix = np.random.default_rng(1).normal(0.2, 0.1, (2, SR)).astype(np.float32)
    stems = denote.separate(mix, sr=SR, segment_s=0.3, overlap_s=0.05).stems
    assert np.allclose(sum(stems.values()), mix, atol=1e-5)


def test_separate_does_not_modify_the_callers_array(fake_model):
    mix = np.random.default_rng(2).normal(0.2, 0.1, (2, SR)).astype(np.float32)
    before = mix.copy()
    denote.separate(mix, sr=SR, segment_s=0.3, overlap_s=0.05)
    assert np.array_equal(mix, before)


@pytest.mark.parametrize("overlap_s", [0.0, 0.0001])
def test_no_chunk_is_shorter_than_the_segment(fake_model, overlap_s):
    # one segment plus 40 frames: the old loop left a tail chunk the model rejects
    mix = np.random.default_rng(3).normal(0, 0.1, (2, SR + 40)).astype(np.float32)
    result = denote.separate(mix, sr=SR, segment_s=1.0, overlap_s=overlap_s)
    assert fake_model.chunks == [SR, SR]
    assert np.allclose(result.stems["bass"], _expected(mix, 0.2), atol=1e-5)


@pytest.mark.parametrize("frames", [3, 441, 768, 1025, 1280, 1281])
def test_inputs_shorter_than_the_model_minimum_are_padded(fake_model, frames):
    mix = np.random.default_rng(4).normal(0, 0.1, (2, frames)).astype(np.float32)
    result = denote.separate(mix, sr=SR)
    assert result.stems["vocals"].shape == (2, frames)
    assert np.allclose(result.stems["vocals"], _expected(mix, 0.4), atol=1e-5)


def test_separate_takes_mono_and_frames_first(fake_model):
    mono = np.sin(np.linspace(0, 100, SR)).astype(np.float32)
    assert denote.separate(mono, sr=SR).stems["vocals"].shape == (2, SR)
    frames_first = np.stack([mono, mono], axis=1)  # (frames, channels), as soundfile
    assert denote.separate(frames_first, sr=SR).stems["bass"].shape == (2, SR)


@pytest.mark.parametrize("layout", ["channels_first", "frames_first"])
def test_separate_takes_torch_tensors(fake_model, layout):
    mix = np.random.default_rng(5).normal(0, 0.1, (2, 22050)).astype(np.float32)
    t = torch.from_numpy(mix if layout == "channels_first" else mix.T.copy())
    result = denote.separate(t, sr=22050)  # resampled along time: 1 s at 44.1 kHz
    assert result.stems["drums"].shape == (2, SR)
    expected = denote.separate(mix, sr=22050).stems["drums"]
    assert np.allclose(result.stems["drums"], expected, atol=1e-6)


def test_separate_resamples_to_the_model_rate(fake_model):
    mono = np.zeros(22050, dtype=np.float32)
    result = denote.separate(np.stack([mono, mono], axis=1), sr=22050)
    assert result.stems["drums"].shape == (2, SR)  # 1 s at 44.1 kHz, along time


def test_sources_keeps_only_the_stems_named(fake_model):
    mix = np.random.default_rng(6).normal(0.1, 0.1, (2, SR)).astype(np.float32)
    full = denote.separate(mix, sr=SR, segment_s=0.4, overlap_s=0.1)
    voice = denote.separate(
        mix, sr=SR, segment_s=0.4, overlap_s=0.1, sources=("vocals",)
    )
    assert list(voice.stems) == ["vocals"]
    assert np.array_equal(voice.stems["vocals"], full.stems["vocals"])
    with pytest.raises(ValueError, match="guitar"):
        denote.separate(mix, sr=SR, sources=("guitar",))


def test_memory_peak_is_the_output_plus_small_change(fake_model):
    # output 4 x 2 x N float32; normalized input 2 x N; weights N. Out-of-place
    # arithmetic on the output would add at least one more output-sized array.
    n = 10 * SR
    mix = np.random.default_rng(7).normal(0, 0.1, (2, n)).astype(np.float32)
    out_bytes = 4 * 2 * n * 4
    tracemalloc.start()
    try:
        denote.separate(mix, sr=SR, segment_s=0.1, overlap_s=0.01)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * out_bytes, peak / out_bytes


def test_separate_rejects_a_segment_no_longer_than_its_overlap(fake_model):
    with pytest.raises(ValueError, match="overlap"):
        denote.separate(np.zeros(SR), sr=SR, segment_s=1.0, overlap_s=1.0)
    with pytest.raises(ValueError, match="overlap"):  # equal once in frames
        denote.separate(np.zeros(SR), sr=SR, segment_s=1.00001, overlap_s=1.0)


def test_unknown_kwargs_get_denotes_usual_warning(fake_model):
    with pytest.warns(UserWarning, match="segment"):
        denote.separate(np.zeros((2, SR), dtype=np.float32), sr=SR, segment=5)


def test_bundle_and_device_reach_the_loader(monkeypatch):
    calls = []

    def load(bundle, device):
        calls.append((bundle, device))
        return _FakeModel(), SR

    monkeypatch.setattr(hd, "_load", load)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # bundle is a known parameter: no warning
        denote.separate(np.zeros((2, SR), dtype=np.float32), sr=SR)
        denote.separate(
            np.zeros((2, SR), dtype=np.float32), sr=SR, bundle="HDEMUCS_HIGH_MUSDB"
        )
    assert calls == [("HDEMUCS_HIGH_MUSDB_PLUS", "cpu"), ("HDEMUCS_HIGH_MUSDB", "cpu")]


def test_separation_result_mono_names_its_stems():
    r = SeparationResult(stems={"vocals": np.ones((2, 4))}, sr=SR)
    with pytest.raises(KeyError, match="vocals"):
        r.mono("drums")


def test_separate_is_the_default_for_its_task():
    assert denote.get_default_backend("separate") == "torchaudio_hdemucs"
