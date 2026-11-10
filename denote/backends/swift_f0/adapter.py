"""Adapter for SwiftF0 — fast CNN pitch estimation on CPU (ONNX)."""

from functools import lru_cache

import numpy as np

from denote.base import PitchResult
from denote.util import load_audio, peak_normalize
from denote.translation import make_kwargs_translator

#: SwiftF0's model rate; audio is resampled to it here (so no soxr is needed).
SWIFT_F0_SR = 16000


@lru_cache(maxsize=2)
def _load(threads):
    """A ``swift_f0.SwiftF0`` detector (builds an ONNX session: reuse it)."""
    from swift_f0 import SwiftF0

    return SwiftF0(threads=threads)


class Adapter:
    """SwiftF0 adapter for monophonic pitch/F0 estimation."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def get_pitch(self, audio, *, sr=None, **kwargs):
        """Estimate pitch from audio with SwiftF0.

        Args:
            audio: File path, numpy array or torch tensor.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters: ``min_frequency``, ``max_frequency``
                (clipped to the model's 46.875-2093.75 Hz), ``confidence_threshold``
                (default 0.5, SwiftF0's calibrated voicing threshold), ``threads``,
                ``normalize`` (default True: peak-normalise first, since SwiftF0
                calls quiet input unvoiced; frames peaking below 1e-3 are forced
                unvoiced. False passes the level through).

        Returns:
            PitchResult with .times, .frequencies (NaN where unvoiced) and
            .confidence; ``.raw`` is SwiftF0's own result (with loudness_db).
        """
        y, actual_sr = load_audio(audio, sr=sr, mono=True, target_sr=SWIFT_F0_SR)
        opts = self._translate(**kwargs)
        threshold = opts.pop("confidence_threshold")
        detector = _load(opts.pop("threads"))
        y = peak_normalize(y) if opts.pop("normalize") else np.asarray(y, np.float32)
        result = detector.detect(
            y,
            actual_sr,
            fmin=opts.get("fmin"),
            fmax=opts.get("fmax"),
        )
        frequencies = result.pitch_hz.copy()
        frequencies[result.confidence < threshold] = np.nan
        return PitchResult(
            times=result.timestamps,
            frequencies=frequencies,
            confidence=result.confidence,
            raw=result,
            backend="swift_f0",
        )
