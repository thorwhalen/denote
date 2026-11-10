"""Adapter for key estimation from librosa chroma and published key profiles."""

import numpy as np

from denote.keys import key_from_chroma
from denote.translation import make_kwargs_translator
from denote.util import DFLT_CHROMA_HOP, cqt_chroma, load_audio


def chroma_stft(y, sr):
    """STFT chromagram: faster than CQT, coarser in the bass."""
    import librosa

    return librosa.feature.chroma_stft(y=y, sr=sr, hop_length=DFLT_CHROMA_HOP)


def chroma_cens(y, sr):
    """CENS chromagram: CQT chroma smoothed and quantized, robust to timbre."""
    import librosa

    return librosa.feature.chroma_cens(y=y, sr=sr, hop_length=DFLT_CHROMA_HOP)


# 'cqt' is denote.util.cqt_chroma, the same call align uses
CHROMA_FUNCTIONS = {"cqt": cqt_chroma, "stft": chroma_stft, "cens": chroma_cens}
WEIGHTINGS = ("energy", "uniform")


def _chroma_function(chroma):
    if callable(chroma):
        return chroma
    if chroma not in CHROMA_FUNCTIONS:
        raise ValueError(
            f"unknown chroma {chroma!r}; pass one of {sorted(CHROMA_FUNCTIONS)} "
            "or a callable (y, sr) -> (12, frames)"
        )
    return CHROMA_FUNCTIONS[chroma]


def frame_rms(y, n_frames, *, hop_length=DFLT_CHROMA_HOP):
    """RMS per frame, resampled to ``n_frames`` (so any chroma callable can use it)."""
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if rms.size == n_frames:
        return rms
    return np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, rms.size), rms)


def _check_level(y, sr, *, min_rms, min_duration_s):
    """Refuse audio too short or too quiet to imply a key (``ValueError``)."""
    duration = y.size / sr
    if duration < min_duration_s:
        raise ValueError(
            f"audio is {duration:.3f} s long; key estimation needs at least "
            f"min_duration_s={min_duration_s:g} s"
        )
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    if rms < min_rms:
        raise ValueError(
            f"audio RMS {rms:.2g} is below min_rms={min_rms:g}: too quiet "
            "(silence or faint noise) to estimate a key"
        )
    return rms, duration


class Adapter:
    """Key estimation: weighted mean chroma correlated with 24 rotated key profiles."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def get_key(self, audio, *, sr=None, **kwargs):
        """Estimate the key (tonic + mode) of a whole recording.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: ``profile`` (a name in ``denote.keys.KEY_PROFILES``, default
                ``"krumhansl"``, or a ``(major, minor)`` pair of 12 weights),
                ``chroma`` (``"cqt"`` default, ``"stft"``, ``"cens"``, or a callable
                ``(y, sr) -> (12, frames)``), ``audio_sr`` (rate the audio is
                resampled to first, default 22050), ``weighting`` (``"energy"``
                default: frames weighted by RMS; or ``"uniform"``), ``min_rms``
                and ``min_duration_s`` (refuse quieter or shorter audio).

        Returns:
            KeyResult; ``confidence`` is the winning key's Pearson correlation,
            ``margin`` its lead over the runner-up, ``scores`` all 24 keys best
            first, ``raw`` the averaged chroma, the RMS and the duration.

        Raises:
            ValueError: audio shorter than ``min_duration_s`` or quieter than
                ``min_rms``, or an unknown ``chroma``/``weighting``/``profile``.
        """
        native = self._translate(**kwargs)
        weighting = native["weighting"]
        if weighting not in WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {WEIGHTINGS}, not {weighting!r}"
            )
        chroma_fn = _chroma_function(native["chroma"])
        y, rate = load_audio(audio, sr=sr, mono=True, target_sr=native["audio_sr"])
        rms, duration = _check_level(
            y,
            rate,
            min_rms=native["min_rms"],
            min_duration_s=native["min_duration_s"],
        )
        chromagram = np.asarray(chroma_fn(y, rate))
        weights = None
        if weighting == "energy" and chromagram.ndim == 2:
            weights = frame_rms(y, chromagram.shape[1])
        result = key_from_chroma(
            chromagram,
            profile=native["profile"],
            weights=weights,
            backend=self._config["name"],
        )
        result.raw.update(rms=rms, duration_s=duration, sr=rate, weighting=weighting)
        return result
