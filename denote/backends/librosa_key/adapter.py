"""Adapter for key estimation from librosa chroma and published key profiles."""

from denote.keys import key_from_chroma
from denote.translation import make_kwargs_translator
from denote.util import load_audio


def chroma_cqt(y, sr):
    """CQT chromagram (librosa defaults: tuning estimated, hop 512)."""
    import librosa

    return librosa.feature.chroma_cqt(y=y, sr=sr)


def chroma_stft(y, sr):
    """STFT chromagram: faster than CQT, coarser in the bass."""
    import librosa

    return librosa.feature.chroma_stft(y=y, sr=sr)


def chroma_cens(y, sr):
    """CENS chromagram: CQT chroma smoothed and quantized, robust to timbre."""
    import librosa

    return librosa.feature.chroma_cens(y=y, sr=sr)


CHROMA_FUNCTIONS = {"cqt": chroma_cqt, "stft": chroma_stft, "cens": chroma_cens}


def _chroma_function(chroma):
    if callable(chroma):
        return chroma
    if chroma not in CHROMA_FUNCTIONS:
        raise ValueError(
            f"unknown chroma {chroma!r}; pass one of {sorted(CHROMA_FUNCTIONS)} "
            "or a callable (y, sr) -> (12, frames)"
        )
    return CHROMA_FUNCTIONS[chroma]


class Adapter:
    """Key estimation: mean chroma correlated with 24 rotated key profiles."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def get_key(self, audio, *, sr=None, **kwargs):
        """Estimate the key (tonic + mode) of a whole recording.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: ``profile`` (a name in ``denote.keys.KEY_PROFILES``, default
                ``"krumhansl"``, or a ``(major, minor)`` pair of 12 weights) and
                ``chroma`` (``"cqt"`` default, ``"stft"``, ``"cens"``, or a callable
                ``(y, sr) -> (12, frames)``).

        Returns:
            KeyResult; ``confidence`` is the winning key's Pearson correlation,
            ``margin`` its lead over the runner-up, ``scores`` all 24 keys best
            first, ``raw["chroma"]`` the mean chroma.
        """
        y, actual_sr = load_audio(audio, sr=sr, mono=True)
        native = self._translate(sr=actual_sr, **kwargs)
        chromagram = _chroma_function(native["chroma"])(y, actual_sr)
        return key_from_chroma(
            chromagram, profile=native["profile"], backend=self._config["name"]
        )
