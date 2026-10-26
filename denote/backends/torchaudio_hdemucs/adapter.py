"""Adapter for torchaudio's Hybrid Demucs (``HDEMUCS_HIGH_MUSDB_PLUS``)."""

from functools import lru_cache

import numpy as np

from denote.base import SeparationResult
from denote.util import load_audio

#: Sample rate of the HDEMUCS_HIGH_MUSDB_PLUS bundle (checked when it loads).
HDEMUCS_SR = 44100


@lru_cache(maxsize=2)
def _model(device: str):
    import torchaudio

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    assert bundle.sample_rate == HDEMUCS_SR, bundle.sample_rate
    return bundle.get_model().to(device).eval()


def overlap_add_weights(n: int, start: int, end: int, overlap: int) -> np.ndarray:
    """Weights of samples ``start:end`` of an ``n``-sample signal cut in chunks that overlap by
    ``overlap``: linear ramps where a chunk meets its neighbours, so the weights of
    overlapping chunks sum to one."""
    w = np.ones(end - start)
    ramp = np.linspace(0, 1, min(overlap, end - start) + 2)[1:-1]
    if start > 0:
        w[: len(ramp)] = ramp
    if end < n:
        w[-len(ramp) :] = ramp[::-1]
    return w


class Adapter:
    """Hybrid Demucs: drums, bass, other, vocals from a stereo mix."""

    def __init__(self, config: dict):
        self._config = config

    def separate(
        self,
        audio,
        *,
        sr=None,
        segment_s: float = 10.0,
        overlap_s: float = 1.0,
        device: str = "cpu",
    ) -> SeparationResult:
        """Separate a mix into four stems.

        Args:
            audio: File path or numpy array (mono, ``(frames, channels)`` or
                ``(channels, frames)``); mono is duplicated to stereo.
            sr: Sample rate (required if audio is an array).
            segment_s: Chunk length; memory grows with it.
            overlap_s: Crossfade between chunks; must be shorter than ``segment_s``.
            device: Torch device.

        Returns:
            SeparationResult with ``(2, frames)`` stems at 44.1 kHz; ``.vocals`` is mono.
        """
        seg, ov = int(segment_s * HDEMUCS_SR), int(overlap_s * HDEMUCS_SR)
        if seg <= ov:  # in samples: the chunk loop would never advance
            raise ValueError(
                f"segment_s ({segment_s}) must be longer than overlap_s ({overlap_s})"
            )
        import torch

        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            if (
                audio.shape[0] > audio.shape[1]
            ):  # (frames, channels): resampling is along the last axis
                audio = audio.T
        y, _ = load_audio(audio, sr=sr, mono=False, target_sr=HDEMUCS_SR)
        x = np.atleast_2d(np.asarray(y, dtype=np.float32))
        x = np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2]
        mean, std = x.mean(), x.std() + 1e-8
        x = torch.from_numpy((x - mean) / std)

        model = _model(device)
        n = x.shape[1]
        out = np.zeros((len(model.sources), 2, n), dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)
        start = 0
        with torch.no_grad():
            while start < n:
                end = min(n, start + seg)
                est = model(x[None, :, start:end].to(device))[0].cpu().numpy()
                w = overlap_add_weights(n, start, end, ov).astype(np.float32)
                out[:, :, start:end] += est * w
                weight[start:end] += w
                if end == n:
                    break
                start = end - ov
        stems = out / np.maximum(weight, 1e-8) * std + mean
        return SeparationResult(
            stems=dict(zip(model.sources, stems)),
            sr=HDEMUCS_SR,
            backend="torchaudio_hdemucs",
        )
