"""Align a symbolic score to a recording of the same music (audio-to-score sync).

:func:`align_score` answers "which moment of the score is playing at time *t*
of this recording?" (and the reverse), for a score and a performance that
differ in tempo, rubato, intro length, even key. It is the bridge you need to
cut a score and a recording at the same musical point, to compare several
scores of a piece against one recording, or to put score positions on a
recording's timeline.

Method: chroma features on both sides (the score's piano roll folded to pitch
classes, no synthesis needed; the audio's constant-Q chroma), cosine distance,
and dynamic time warping (``librosa.sequence.dtw``). By default the recording
is matched as a *subsequence* of the score, so a clip of the first minute finds
its place inside a full-length score. ``transpose="auto"`` also tries all
twelve transpositions of the score and keeps the best, because arrangements
are often in another key.

The result's ``cost`` (mean cosine distance along the path, 0 = identical
pitch content, 1 = unrelated) is comparable across scores aligned to the same
recording, which makes it a ranking of how faithful each score is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from denote.base import AudioInput
from denote.util import load_audio

#: Default analysis rate (frames per second) for both score and audio.
DFLT_FRAME_RATE = 10.0
#: Default sample rate the audio is resampled to before chroma extraction.
DFLT_AUDIO_SR = 22050
#: DTW steps (audio frames, score frames). Without (1, 0)/(0, 1) steps the path
#: slope is bounded to [1/2, 2]: the performance may run at half to double the
#: score's tempo (strictly inside that range at the end points), but cannot sit
#: on one score frame for a whole clip, which is
#: the degenerate answer unconstrained subsequence DTW gives.
_STEPS = np.array([[1, 1], [1, 2], [2, 1]])
_STEP_WEIGHTS = np.array([1.0, 1.0, 1.0])


@dataclass
class ScoreAlignment:
    """A monotone time map between a score's timeline and a recording's.

    Attributes:
        score_times: Score times in seconds, increasing (one per path step).
        audio_times: The matching recording times in seconds.
        cost: Mean cosine distance along the path (lower is a closer match).
        transpose: Semitones the score was shifted to match the recording.
        raw: Backend detail (the warping path in frames, the frame rate).
    """

    score_times: np.ndarray
    audio_times: np.ndarray
    cost: float
    transpose: int = 0
    raw: Any = field(default=None, repr=False)

    def score_to_audio(self, t):
        """Recording time(s) at which score time(s) ``t`` sound."""
        return np.interp(t, self.score_times, self.audio_times)

    def audio_to_score(self, t):
        """Score time(s) sounding at recording time(s) ``t``."""
        return np.interp(t, self.audio_times, self.score_times)

    @property
    def score_span(self) -> tuple:
        """``(first, last)`` score times covered by the recording."""
        return float(self.score_times[0]), float(self.score_times[-1])


def _to_pretty_midi(score):
    if type(score).__name__ == "PrettyMIDI":
        return score
    if isinstance(score, (str, Path)) and Path(score).suffix.lower() in (".mid", ".midi"):
        import pretty_midi

        return pretty_midi.PrettyMIDI(str(score))
    try:
        from audiate import to_pretty_midi  # MusicXML, ABC, kern, music21, ...
    except ImportError as exc:
        raise TypeError(
            "align_score takes a pretty_midi.PrettyMIDI or a .mid path; for "
            "MusicXML/ABC/... install audiate (pip install 'audiate[symbolic]')."
        ) from exc
    return to_pretty_midi(score)


def _normalize(chroma: np.ndarray) -> np.ndarray:
    """Unit-norm columns; a silent frame becomes the flat (uniform) chroma.

    A zero column would make the cosine distance undefined (NaN) and break the
    DTW, and silence genuinely carries no pitch-class preference.
    """
    chroma = np.asarray(chroma, dtype=float).copy()
    norms = np.linalg.norm(chroma, axis=0)
    silent = norms < 1e-9
    chroma[:, silent] = 1.0
    norms[silent] = np.sqrt(chroma.shape[0])
    return chroma / norms


def score_chroma(score, *, frame_rate: float = DFLT_FRAME_RATE) -> np.ndarray:
    """L2-normalized 12 x frames chroma of a score (drums excluded)."""
    pm = _to_pretty_midi(score)
    return _normalize(pm.get_chroma(fs=frame_rate))


def audio_chroma(
    audio: AudioInput,
    *,
    sr: Optional[int] = None,
    frame_rate: float = DFLT_FRAME_RATE,
    audio_sr: int = DFLT_AUDIO_SR,
) -> np.ndarray:
    """L2-normalized 12 x frames constant-Q chroma of a recording."""
    import librosa

    y, rate = load_audio(audio, sr=sr, mono=True, target_sr=audio_sr)
    hop = int(round(rate / frame_rate))
    return _normalize(librosa.feature.chroma_cqt(y=y, sr=rate, hop_length=hop))


def align_score(
    score,
    audio: AudioInput,
    *,
    sr: Optional[int] = None,
    subsequence: bool = True,
    transpose: Union[int, str] = 0,
    frame_rate: float = DFLT_FRAME_RATE,
    audio_sr: int = DFLT_AUDIO_SR,
) -> ScoreAlignment:
    """Align a score to a recording of (part of) the same music.

    Args:
        score: A ``pretty_midi.PrettyMIDI`` or a ``.mid`` path (other score
            formats when ``audiate`` is installed).
        audio: A recording: a path, or an array with ``sr``.
        sr: Sample rate when ``audio`` is an array.
        subsequence: Match the recording anywhere inside the score (default),
            e.g. a one-minute clip against a full score. ``False`` forces the
            two to start and end together.
        transpose: Semitones to shift the score's pitch classes, or ``"auto"``
            to try all twelve and keep the lowest-cost one.
        frame_rate: Analysis frames per second for both sides.
        audio_sr: Rate the audio is resampled to before analysis.

    Returns:
        A :class:`ScoreAlignment`.

    Example::

        a = denote.align_score("theme.mid", "recording_first_minute.wav")
        a.audio_to_score(60.0)      # where the score is when the clip ends
        a.cost                      # compare scores of the same piece
    """
    S = score_chroma(score, frame_rate=frame_rate)
    A = audio_chroma(audio, sr=sr, frame_rate=frame_rate, audio_sr=audio_sr)
    return _align_chroma(
        A, S, subsequence=subsequence, transpose=transpose, frame_rate=frame_rate
    )


def align_audio(
    audio: AudioInput,
    reference: AudioInput,
    *,
    sr: Optional[int] = None,
    reference_sr: Optional[int] = None,
    subsequence: bool = False,
    transpose: Union[int, str] = 0,
    frame_rate: float = DFLT_FRAME_RATE,
    audio_sr: int = DFLT_AUDIO_SR,
) -> ScoreAlignment:
    """Align two recordings of the same music (a cover, a render, a remix).

    The same chroma DTW as :func:`align_score`, with a recording in the
    score's place: ``score_times`` are times in ``reference``. The ``cost`` is
    a measure of how closely ``audio`` follows ``reference``'s harmony and
    melody, tolerant of tempo differences and blind to timbre, which makes it
    a fidelity score for generated versions of a piece.

    Args:
        audio: The recording to place on the reference's timeline.
        reference: The reference recording.
        sr: Sample rate when ``audio`` is an array.
        reference_sr: Sample rate when ``reference`` is an array.
        subsequence: Match ``audio`` anywhere inside ``reference`` (default
            ``False``: both start and end together).
        transpose: Semitones, or ``"auto"`` to try all twelve.
        frame_rate: Analysis frames per second.
        audio_sr: Rate both are resampled to before analysis.

    Returns:
        A :class:`ScoreAlignment` (``score_*`` refers to ``reference``).
    """
    R = audio_chroma(reference, sr=reference_sr, frame_rate=frame_rate, audio_sr=audio_sr)
    A = audio_chroma(audio, sr=sr, frame_rate=frame_rate, audio_sr=audio_sr)
    return _align_chroma(
        A, R, subsequence=subsequence, transpose=transpose, frame_rate=frame_rate
    )


def _align_chroma(A, S, *, subsequence, transpose, frame_rate) -> ScoreAlignment:
    """DTW of query chroma ``A`` against reference chroma ``S``."""
    import librosa

    if S.shape[1] == 0 or A.shape[1] == 0:
        raise ValueError("score or audio is empty")
    n, m = A.shape[1], S.shape[1]
    if min(n, m) < 2:
        raise ValueError(
            f"too short to align: {n} and {m} frames at {frame_rate} frames/s"
        )
    shifts = range(12) if transpose == "auto" else [int(transpose) % 12]

    best = None
    for k in shifts:
        Sk = np.roll(S, k, axis=0)
        # librosa matches X as a subsequence of Y when subseq=True.
        try:
            _, wp = librosa.sequence.dtw(
                X=A,
                Y=Sk,
                metric="cosine",
                subseq=subsequence,
                step_sizes_sigma=_STEPS,
                weights_mul=_STEP_WEIGHTS,
            )
        except librosa.util.exceptions.ParameterError as exc:
            raise ValueError(
                f"No alignment within the allowed tempo range (the recording at "
                f"strictly between half and double the reference's speed): "
                f"{n / frame_rate:.1f}s of query against {m / frame_rate:.1f}s of "
                f"reference, subsequence={subsequence}. Crop the longer one to "
                f"the shared passage, or pass subsequence=True to find a clip "
                f"inside a longer reference."
            ) from exc
        wp = wp[::-1]  # path runs end -> start
        # Mean cosine distance along the path (columns are unit-norm). Not
        # read off the accumulated matrix: librosa may transpose it when the
        # query is longer than the reference.
        cost = float(np.mean(1.0 - np.sum(A[:, wp[:, 0]] * Sk[:, wp[:, 1]], axis=0)))
        if best is None or cost < best[0]:
            best = (cost, k, wp)
    cost, k, wp = best

    audio_frames, score_frames = wp[:, 0], wp[:, 1]
    # Keep one score time per audio frame so the map is strictly usable both ways.
    _, first = np.unique(audio_frames, return_index=True)
    audio_t = audio_frames[first] / frame_rate
    score_t = np.maximum.accumulate(score_frames[first] / frame_rate)
    return ScoreAlignment(
        score_times=score_t,
        audio_times=audio_t,
        cost=cost,
        transpose=k if k <= 6 else k - 12,
        raw={"path": wp, "frame_rate": frame_rate},
    )
