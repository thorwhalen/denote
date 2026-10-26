"""Adapter for torchaudio's Hybrid Demucs bundles (``HDEMUCS_HIGH_MUSDB_PLUS`` by default)."""

from functools import lru_cache

import numpy as np

from denote.base import SeparationResult
from denote.translation import make_kwargs_translator
from denote.util import chunked_apply, load_audio

#: Shortest input HDemucs accepts for sure. Measured on the torchaudio bundles
#: (nfft 4096, hop 1024): lengths 769..1024 and >= 1281 run, 1..768 and 1025..1280
#: fail inside ``_spec`` ("Spectrogram's last dimension must be ..."), because its
#: reflect padding falls back to extra zero padding that changes the frame count.
#: Shorter chunks are zero-padded to this and cropped back.
MIN_FRAMES = 1281

#: Added to the mix's std before normalizing, so silence does not divide by zero.
NORM_EPS = 1e-8


@lru_cache(maxsize=2)
def _load(bundle: str, device: str):
    """``(model, sample_rate)`` of the torchaudio.pipelines bundle named ``bundle``."""
    import torchaudio

    b = getattr(torchaudio.pipelines, bundle, None)
    if b is None or not hasattr(b, "get_model"):
        raise ValueError(
            f"{bundle!r} is not a torchaudio.pipelines bundle; try "
            "'HDEMUCS_HIGH_MUSDB_PLUS' or 'HDEMUCS_HIGH_MUSDB'"
        )
    return b.get_model().to(device).eval(), b.sample_rate


def _as_numpy(audio):
    """Torch tensors to numpy (without importing torch); anything else as is."""
    if hasattr(audio, "detach") and hasattr(audio, "numpy"):
        return audio.detach().cpu().numpy()
    return audio


class Adapter:
    """Hybrid Demucs: drums, bass, other, vocals from a stereo mix."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def separate(self, audio, *, sr=None, **kwargs) -> SeparationResult:
        """Separate a mix into stems.

        Args:
            audio: File path, numpy array or torch tensor (mono, ``(frames, channels)``
                or ``(channels, frames)``); mono is duplicated to stereo.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters (see ``config.param_map``):
                ``segment_s`` (chunk length; memory per chunk grows with it),
                ``overlap_s`` (crossfade between chunks, 0 allowed, shorter than
                ``segment_s``), ``device``, ``bundle`` (a torchaudio.pipelines name:
                ``HDEMUCS_HIGH_MUSDB_PLUS`` or ``HDEMUCS_HIGH_MUSDB``), ``sources``
                (stem names to keep, default all; the others are never stored).

        Returns:
            SeparationResult with ``(2, frames)`` stems at the bundle's rate (44.1 kHz).
            All four stems sum to the mix; ``.vocals`` is mono.
        """
        opts = self._translate(**kwargs)
        segment_s, overlap_s = opts["segment_s"], opts["overlap_s"]
        if not 0 <= overlap_s < segment_s:
            raise ValueError(
                f"need 0 <= overlap_s < segment_s; got overlap_s={overlap_s}, "
                f"segment_s={segment_s}"
            )
        import torch

        model, model_sr = _load(opts["bundle"], opts["device"])
        names = list(model.sources)
        keep = names if opts["sources"] is None else list(opts["sources"])
        if unknown := set(keep) - set(names):
            raise ValueError(
                f"unknown sources {sorted(unknown)}; this model has {names}"
            )
        idx = [names.index(s) for s in keep]

        audio = _as_numpy(audio)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            if (
                audio.shape[0] > audio.shape[1]
            ):  # (frames, channels): resample along time
                audio = audio.T
        y, _ = load_audio(audio, sr=sr, mono=False, target_sr=model_sr)
        x = np.atleast_2d(np.asarray(y, dtype=np.float32))
        del y  # x - mean below is the only full-size copy of the input we keep
        x = np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2]
        mean, std = x.mean(), x.std() + NORM_EPS
        x = x - mean  # a copy: never normalize the caller's array in place
        x /= std

        def run(chunk):  # (2, m) -> (len(keep), 2, m)
            m = chunk.shape[-1]
            if m < MIN_FRAMES:
                chunk = np.pad(chunk, ((0, 0), (0, MIN_FRAMES - m)))
            t = torch.from_numpy(np.ascontiguousarray(chunk))[None].to(opts["device"])
            with torch.no_grad():
                est = model(t)[0, idx]
            return est.cpu().numpy()[..., :m]

        seg, ov = int(segment_s * model_sr), int(overlap_s * model_sr)
        stems = chunked_apply(run, x, segment=seg, overlap=ov)
        stems *= std  # in place: the output is the only full-size array kept
        stems += mean / len(names)  # each stem's share of the mix's mean
        return SeparationResult(
            stems=dict(zip(keep, stems)),
            sr=model_sr,
            backend="torchaudio_hdemucs",
        )
