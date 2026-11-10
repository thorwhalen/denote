"""Adapter for PESTO — self-supervised pitch estimation (Sony CSL)."""

from functools import lru_cache

import numpy as np

from denote.base import PitchResult
from denote.util import load_audio, peak_normalize
from denote.translation import make_kwargs_translator


def frame_period(step_size_ms: float, sr: int) -> float:
    """Seconds between PESTO's frames for a ``step_size_ms`` step at ``sr``.

    PESTO rounds the step to whole samples (``int(step * sr / 1000 + 0.5)``), so
    the true period is that hop over ``sr``, not ``step_size_ms / 1000``: at
    22050 Hz a 10 ms step is 221 samples (10.023 ms), and times built from the
    nominal step drift 0.23 % (0.7 s at 5 minutes).
    """
    return int(step_size_ms * sr / 1000 + 0.5) / sr


@lru_cache(maxsize=4)
def _load(model_name: str, step_size: float, sr: int, reduction: str, device: str):
    """A PESTO model for one (checkpoint, step, rate, reduction, device).

    Everything the model holds as state is in the key, so a cached model is never
    mutated by a later call with other options.
    """
    from pesto import load_model

    model = load_model(model_name, step_size, sampling_rate=sr).to(device).eval()
    model.reduction = reduction
    return model


def _predict(model, y: np.ndarray, sr: int, device: str) -> dict:
    """Run ``model`` on mono ``y``: numpy ``pitch`` (Hz), ``confidence``, ``volume``,
    ``activations``. The only torch-touching step (the seam tests replace)."""
    import torch

    x = torch.as_tensor(y, device=device)
    with torch.inference_mode():
        pitch, confidence, volume, activations = model(
            x, sr=sr, convert_to_freq=True, return_activations=True
        )
    return {
        "pitch": pitch.cpu().numpy().astype(float),
        "confidence": confidence.cpu().numpy().astype(float),
        "volume": volume.cpu().numpy(),
        "activations": activations.cpu().numpy(),
    }


class Adapter:
    """PESTO adapter for monophonic pitch/F0 estimation."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def get_pitch(self, audio, *, sr=None, **kwargs):
        """Estimate pitch from audio with PESTO.

        Args:
            audio: File path, numpy array or torch tensor.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters: ``step_size_ms`` (default 10),
                ``model`` (checkpoint name or path), ``reduction``, ``device``,
                ``confidence_threshold`` (default 0.5), ``normalize`` (default
                True: peak-normalise first, since PESTO's confidence falls with
                level and no frame is voiced at -30 dBFS; False passes the level
                through).

        Returns:
            PitchResult with .times, .frequencies (NaN where unvoiced) and
            .confidence; ``.raw`` holds the pitch, volume and activations.
        """
        y, actual_sr = load_audio(audio, sr=sr, mono=True)
        opts = self._translate(**kwargs)
        y = peak_normalize(y) if opts["normalize"] else np.asarray(y, np.float32)
        model = _load(
            opts["model_name"],
            opts["step_size"],
            actual_sr,
            opts["reduction"],
            opts["device"],
        )
        out = _predict(model, y, actual_sr, opts["device"])
        pitch, confidence = out["pitch"], out["confidence"]
        times = np.arange(len(pitch)) * frame_period(opts["step_size"], actual_sr)
        frequencies = pitch.copy()
        frequencies[confidence < opts["confidence_threshold"]] = np.nan
        return PitchResult(
            times=times,
            frequencies=frequencies,
            confidence=confidence,
            raw=out,
            backend="pesto",
        )
