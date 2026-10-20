"""Compare a version of a piece with a reference: harmony and sound, on the same moments (denote#3).

:func:`~denote.align.align_audio` measures how closely a version follows the
reference's harmony and melody (chroma DTW), and is blind to timbre on purpose.
:func:`compare_audio` adds the other half, how close the *sound* is, and keeps
both on one footing: the two recordings are aligned once, and every metric reads
the same warping path, so each compares the same musical moments.

Metrics are a registry (:data:`METRICS`, a ``name -> metric`` mapping you can
extend or replace through ``metrics=``): a metric is a callable taking an
:class:`AlignedPair` and returning a distance (lower is closer). An embedding
metric (CLAP, MERT, a Fréchet-style distance) is one more entry, with no change
to callers.

Built in:

- ``"chroma"``: the alignment's own cost, mean cosine distance between chroma
  frames along the path (0 identical pitch content, 1 unrelated). Measured on an
  orchestral minute, an unrelated piece in the same key and style scored about
  0.16-0.21 against the reference, so it discriminates mainly below about 0.15.
- ``"timbre"``: MFCCs ``c1..c19`` (``c0``, loudness, left out) of both
  recordings, each coefficient z-scored with statistics pooled over both (so a
  global difference in spectral shape is kept, not normalised away), then the
  root-mean-square difference per coefficient, averaged along the path. 0 is the
  same sound; two unrelated timbres are typically around 1. The starter it
  replaces (z-scored per recording) separated a General MIDI render
  (0.46-0.79) from a light audio-to-audio pass of the original (0.41); it is a
  coarse spectral-envelope measure, not a perceptual one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Union

import numpy as np

from denote.align import DFLT_AUDIO_SR, DFLT_FRAME_RATE, ScoreAlignment, align_audio
from denote.base import AudioInput
from denote.util import load_audio

__all__ = [
    "AlignedPair",
    "AudioComparison",
    "Metric",
    "METRICS",
    "DFLT_METRICS",
    "compare_audio",
    "chroma_metric",
    "timbre_metric",
]

#: MFCCs computed for the timbre metric; ``c0`` (loudness) is then dropped.
DFLT_N_MFCC = 20


@dataclass(frozen=True)
class AlignedPair:
    """Two recordings at one rate, and the alignment between them, as metrics see them.

    ``audio`` and ``reference`` are mono samples at ``sr``; ``alignment.raw["path"]``
    holds the warping path as ``(audio frame, reference frame)`` rows at
    ``frame_rate`` frames per second (hop ``hop_length`` samples).
    """

    audio: np.ndarray
    reference: np.ndarray
    sr: int
    frame_rate: float
    alignment: ScoreAlignment

    @property
    def hop_length(self) -> int:
        return int(round(self.sr / self.frame_rate))

    @property
    def path(self) -> np.ndarray:
        return np.asarray(self.alignment.raw["path"])


#: A metric: an aligned pair -> a distance (lower is closer).
Metric = Callable[[AlignedPair], float]


def chroma_metric(pair: AlignedPair) -> float:
    """The alignment's chroma cost: mean cosine distance along the path."""
    return float(pair.alignment.cost)


def timbre_metric(pair: AlignedPair, *, n_mfcc: int = DFLT_N_MFCC) -> float:
    """Mean RMS difference of pooled-z-scored MFCCs ``c1..`` along the path; see the module."""
    import librosa

    def mfcc(y: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(
            y=y, sr=pair.sr, n_mfcc=n_mfcc, hop_length=pair.hop_length
        )[1:]

    A, R = mfcc(pair.audio), mfcc(pair.reference)
    pooled = np.concatenate([A, R], axis=1)
    mean = pooled.mean(axis=1, keepdims=True)
    std = pooled.std(axis=1, keepdims=True)
    std[std < 1e-9] = 1.0  # a constant coefficient carries no difference
    A, R = (A - mean) / std, (R - mean) / std
    path = pair.path
    # Chroma and MFCC frames share the hop; clip in case a frame count differs by one.
    ia = np.minimum(path[:, 0], A.shape[1] - 1)
    ir = np.minimum(path[:, 1], R.shape[1] - 1)
    return float(np.mean(np.sqrt(np.mean((A[:, ia] - R[:, ir]) ** 2, axis=0))))


#: The built-in metrics, by name.
METRICS: Mapping[str, Metric] = MappingProxyType(
    {"chroma": chroma_metric, "timbre": timbre_metric}
)
#: What :func:`compare_audio` computes when ``metrics`` is not given.
DFLT_METRICS = ("chroma", "timbre")


@dataclass(frozen=True)
class AudioComparison:
    """Distances between a version and a reference (lower is closer), and their alignment.

    ``scores`` maps each metric name to its distance; ``comparison["timbre"]``
    reads one.
    """

    scores: Mapping[str, float]
    alignment: ScoreAlignment = field(repr=False)

    def __getitem__(self, name: str) -> float:
        return self.scores[name]


def compare_audio(
    audio: AudioInput,
    reference: AudioInput,
    *,
    metrics: Union[Iterable[str], Mapping[str, Metric]] = DFLT_METRICS,
    registry: Mapping[str, Metric] = METRICS,
    sr: Optional[int] = None,
    reference_sr: Optional[int] = None,
    subsequence: bool = False,
    transpose: Union[int, str] = 0,
    frame_rate: float = DFLT_FRAME_RATE,
    audio_sr: int = DFLT_AUDIO_SR,
) -> AudioComparison:
    """How close ``audio`` is to ``reference``, metric by metric, on the same musical moments.

    Args:
        audio: The version to judge: a path, or an array with ``sr``.
        reference: The reference: a path, or an array with ``reference_sr``.
        metrics: Metric names looked up in ``registry``, or a ``{name: metric}``
            mapping used as is (to add one without registering it).
        registry: Where names are looked up (default :data:`METRICS`).
        subsequence, transpose, frame_rate, audio_sr: As for
            :func:`~denote.align.align_audio`, which is run once and shared.

    Returns:
        An :class:`AudioComparison`.

    Example::

        c = denote.compare_audio("render.wav", "original.wav")
        c["chroma"], c["timbre"]
    """
    if isinstance(metrics, str):
        metrics = (metrics,)
    if isinstance(metrics, Mapping):
        chosen = dict(metrics)
    else:
        names = list(metrics)
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise ValueError(f"unknown metric(s) {unknown}; known: {sorted(registry)}")
        chosen = {n: registry[n] for n in names}
    if not chosen:
        raise ValueError("compare_audio needs at least one metric")
    y, rate = load_audio(audio, sr=sr, mono=True, target_sr=audio_sr)
    y_ref, _ = load_audio(reference, sr=reference_sr, mono=True, target_sr=audio_sr)
    alignment = align_audio(
        y,
        y_ref,
        sr=rate,
        reference_sr=rate,
        subsequence=subsequence,
        transpose=transpose,
        frame_rate=frame_rate,
        audio_sr=audio_sr,
    )
    pair = AlignedPair(y, y_ref, rate, frame_rate, alignment)
    return AudioComparison(
        scores={name: float(metric(pair)) for name, metric in chosen.items()},
        alignment=alignment,
    )
