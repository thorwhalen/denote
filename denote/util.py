"""Audio loading (any layout to ``(channels, frames)``), and chunked overlap-add for models
that take bounded lengths (``chunked_apply``)."""

from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np


def load_audio(
    audio: Union[str, Path, np.ndarray],
    *,
    sr: Optional[int] = None,
    mono: bool = True,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from a file path or validate an existing array.

    Args:
        audio: File path (str/Path) or numpy array of audio samples.
        sr: Sample rate. Required when audio is an array. Ignored for file paths
            (detected automatically).
        mono: If True, convert to mono.
        target_sr: If set, resample to this rate.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        ValueError: If audio is an array and sr is not provided.
        FileNotFoundError: If audio is a path that doesn't exist.
    """
    import librosa

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        y, file_sr = librosa.load(str(path), sr=target_sr, mono=mono)
        return y, target_sr or file_sr

    if isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError(
                "Sample rate (sr) is required when audio is a numpy array."
            )
        y = audio
        if np.issubdtype(y.dtype, np.integer):
            # PCM integers (e.g. int16 from soundfile/scipy): scale to [-1, 1] floats.
            y = y.astype(np.float32) / np.iinfo(y.dtype).max
        elif not np.issubdtype(y.dtype, np.floating):
            y = y.astype(np.float32)
        if y.ndim > 1:
            y = channels_first(y)
        if mono and y.ndim > 1:
            y = librosa.to_mono(np.ascontiguousarray(y))
        if target_sr is not None and target_sr != sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            return y, target_sr
        return y, sr

    # Try torch.Tensor (on any device)
    try:
        y = audio.detach().cpu().numpy()
        if sr is None:
            raise ValueError("Sample rate (sr) is required when audio is a tensor.")
        return load_audio(y, sr=sr, mono=mono, target_sr=target_sr)
    except AttributeError:
        pass

    raise TypeError(
        f"Unsupported audio type: {type(audio)}. "
        "Expected str, Path, np.ndarray, or torch.Tensor."
    )


def channels_first(y: np.ndarray) -> np.ndarray:
    """A multichannel array as ``(channels, frames)``, whichever way it came.

    librosa and torchaudio give ``(channels, frames)``, soundfile
    ``(frames, channels)``. Leading size-1 axes (a batch of one) are dropped. An
    axis of size 2 or less beside a longer one is the channels axis, so a
    one-frame stereo clip stays stereo; otherwise the longer axis is time.
    """
    while y.ndim > 2 and y.shape[0] == 1:
        y = y[0]
    if y.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {y.shape}")
    rows, cols = y.shape
    if cols <= 2 < rows or (rows > cols and not rows <= 2):
        return y.T
    return y


def ensure_file_path(
    audio: Union[str, Path, np.ndarray],
    *,
    sr: Optional[int] = None,
) -> str:
    """Ensure audio is available as a file path.

    If audio is already a path, return it as a string.
    If it's an array, write to a temporary WAV file and return the path.
    """
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return str(path)

    # It's an array — write to temp file
    import tempfile
    import soundfile as sf

    if sr is None:
        raise ValueError("Sample rate (sr) is required when audio is an array.")
    y, actual_sr = load_audio(audio, sr=sr, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, actual_sr)
    return tmp.name


def chunk_bounds(n: int, *, segment: int, overlap: int) -> List[Tuple[int, int]]:
    """``(start, end)`` of the chunks that cover ``n`` frames, ``segment`` long and
    overlapping by at least ``overlap``.

    Every chunk is exactly ``segment`` long (one chunk of ``n`` when ``n <= segment``):
    the last one ends at ``n``, so it may overlap its neighbour by more than
    ``overlap``.

    >>> chunk_bounds(10, segment=4, overlap=1)
    [(0, 4), (3, 7), (6, 10)]
    >>> chunk_bounds(9, segment=4, overlap=0)
    [(0, 4), (4, 8), (5, 9)]
    >>> chunk_bounds(3, segment=4, overlap=1)
    [(0, 3)]
    """
    if not 0 <= overlap < segment:
        raise ValueError(
            f"need 0 <= overlap < segment (in frames); got overlap={overlap}, "
            f"segment={segment}"
        )
    if n <= segment:
        return [(0, n)]
    starts = list(range(0, n - segment, segment - overlap)) + [n - segment]
    return [(s, s + segment) for s in starts]


def chunked_apply(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    segment: int,
    overlap: int,
) -> np.ndarray:
    """Apply ``fn`` to overlapping chunks of ``x`` along its last axis and overlap-add
    the results, so ``fn`` only ever sees ``segment`` frames at a time.

    Args:
        fn: Takes a ``(..., m)`` chunk of ``x`` and returns a ``(..., m)`` array (any
            leading shape, the same for every chunk; ``m`` frames, as given).
        x: Array, typically ``(channels, frames)``.
        segment: Chunk length in frames. Every chunk has this length unless ``x`` is
            shorter (see ``chunk_bounds``).
        overlap: Minimum overlap between neighbouring chunks, in frames (0 allowed).

    Returns:
        ``(..., frames)``: where chunks overlap, a weighted average of their outputs,
        crossfaded linearly over the actual overlap. The output is divided by the
        summed weights, so it is an average even where ramps collide (an overlap
        above half the segment); the ramps alone sum to one wherever exactly two
        chunks overlap and do not collide.

    Only the output, a ``(frames,)`` weight vector and one chunk's temporaries are
    held in memory; the division is done in place.

    >>> x = np.arange(10.0)[None]
    >>> np.allclose(chunked_apply(lambda c: 2 * c, x, segment=4, overlap=1), 2 * x)
    True
    """

    def crossfade(length, fade_in, fade_out, dtype):
        # linear ramps, strictly positive, over the first fade_in and last fade_out
        w = np.ones(length, dtype=dtype)
        if fade_in:
            w[:fade_in] *= np.linspace(0, 1, fade_in + 2, dtype=dtype)[1:-1]
        if fade_out:
            w[length - fade_out :] *= np.linspace(1, 0, fade_out + 2, dtype=dtype)[1:-1]
        return w

    n = x.shape[-1]
    bounds = chunk_bounds(n, segment=segment, overlap=overlap)
    out = weight = None
    for i, (s, e) in enumerate(bounds):
        y = np.asarray(fn(x[..., s:e]))
        if y.shape[-1] != e - s:
            raise ValueError(
                f"fn returned {y.shape[-1]} frames for a {e - s}-frame chunk"
            )
        if out is None:
            dtype = np.result_type(y.dtype, np.float32)
            out = np.zeros(y.shape[:-1] + (n,), dtype=dtype)
            weight = np.zeros(n, dtype=dtype)
        fade_in = bounds[i - 1][1] - s if i > 0 else 0
        fade_out = e - bounds[i + 1][0] if i + 1 < len(bounds) else 0
        w = crossfade(e - s, fade_in, fade_out, out.dtype)
        out[..., s:e] += y * w
        weight[s:e] += w
    out /= weight
    return out
