"""Adapter for Beat This! — transformer beat and downbeat tracking."""

from functools import lru_cache

import numpy as np

from denote.base import BeatResult
from denote.util import load_audio
from denote.translation import make_kwargs_translator

#: The rate Beat This! computes its spectrogram at; audio is resampled to it here.
BEAT_THIS_SR = 22050


@lru_cache(maxsize=2)
def _load(checkpoint_path: str, device: str, dbn: bool, float16: bool):
    """A ``beat_this.inference.Audio2Beats``: ``(signal, sr) -> (beats, downbeats)``."""
    from beat_this.inference import Audio2Beats

    return Audio2Beats(
        checkpoint_path=checkpoint_path, device=device, dbn=dbn, float16=float16
    )


def tempo_from_beats(beats: np.ndarray) -> float | None:
    """BPM from the median inter-beat interval, or ``None`` with fewer than 2 beats."""
    if len(beats) < 2:
        return None
    return 60.0 / float(np.median(np.diff(beats)))


class Adapter:
    """Beat This! adapter: beats and downbeats in seconds."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def get_beats(self, audio, *, sr=None, **kwargs):
        """Track beats and downbeats with Beat This!.

        Args:
            audio: File path, numpy array or torch tensor.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters: ``model`` (checkpoint name, path or
                URL; default 'final0'), ``device``, ``dbn`` (madmom DBN
                post-processing), ``float16``.

        Returns:
            BeatResult with .beats, .downbeats (seconds) and .tempo (BPM from the
            median inter-beat interval; None with fewer than two beats).
        """
        y, actual_sr = load_audio(audio, sr=sr, mono=True, target_sr=BEAT_THIS_SR)
        opts = self._translate(**kwargs)
        tracker = _load(
            opts["checkpoint_path"], opts["device"], opts["dbn"], opts["float16"]
        )
        beats, downbeats = tracker(np.asarray(y, dtype=np.float32), actual_sr)
        beats = np.asarray(beats, dtype=float)
        downbeats = np.asarray(downbeats, dtype=float)
        return BeatResult(
            beats=beats,
            downbeats=downbeats,
            tempo=tempo_from_beats(beats),
            raw={"beats": beats, "downbeats": downbeats, "options": opts},
            backend="beat_this",
        )
