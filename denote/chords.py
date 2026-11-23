"""Model-free chord recognition: spectral-peak chroma, chord templates, a hidden Markov model.

The pieces behind the ``chroma_hmm`` backend, usable on their own:

- ``peaks_chroma(y, sr)``: a ``(frames, 12)`` chroma from spectral peaks, log-compressed
- ``chord_templates(vocabulary)``: unit templates with each tone's overtones folded in
- ``decode_chords(chroma, ...)``: free decoding (Viterbi, or the causal forward filter),
  optionally restricted to a ``vocabulary``
- ``align_sequence(chroma, sequence)``: a left-to-right decode through a known ordered
  chord list (chord-sequence-to-audio alignment: *when* each chord happens)

Labels are Harte syntax (``C:maj``, ``A:min7``, ``N`` for no chord), the vocabulary of
``mir_eval.chord`` and the JAMS chord namespace. All times are seconds.

>>> chord_label(9, "min")
'A:min'
>>> parse_chord_label("Am7")
('A:min7', 9, (0, 3, 7, 10))
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

NO_CHORD = "N"
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: Harte shorthand -> semitone intervals above the root. The SSOT for template tones.
QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
}
_INTERVALS_TO_QUALITY = {iv: q for q, iv in QUALITY_INTERVALS.items()}

DFLT_QUALITIES = ("maj", "min")
#: Cosine a frame must beat to be a chord (measured on GuitarSet: flat between 0.4 and 0.6).
DFLT_NO_CHORD_COS = 0.5
#: Pitch class of harmonics 1..6 above a tone (octave, fifth, octave, third, fifth).
_HARMONIC_SEMITONES = (0, 0, 7, 0, 4, 7)
_SHARPS = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Fb": "E",
    "E#": "F",
    "B#": "C",
}


def chord_label(root: int, quality: str) -> str:
    """Harte label of a pitch class and a quality shorthand."""
    return f"{PITCH_CLASSES[root % 12]}:{quality}"


def _root_pc(name: str) -> int:
    name = name[0].upper() + name[1:]
    name = _SHARPS.get(name, name)
    try:
        return PITCH_CLASSES.index(name)
    except ValueError:
        raise ValueError(f"unknown root {name!r}") from None


def parse_chord_label(label: str) -> Tuple[str, int, Tuple[int, ...]]:
    """``(harte_label, root_pc, intervals)`` of a chord given as Harte (``A:min7``) or in common
    symbols (``Am7``, ``F#m``, ``Bb``; needs ``chordsym``). Raises ``ValueError`` when unreadable."""
    if label == NO_CHORD:
        raise ValueError("'N' is not a chord to template")
    if ":" in label:
        root_name, _, quality = label.partition(":")
        if quality in QUALITY_INTERVALS:
            root = _root_pc(root_name)
            return chord_label(root, quality), root, QUALITY_INTERVALS[quality]
    try:
        import chordsym
    except ImportError:
        raise ValueError(
            f"cannot read chord {label!r}: use Harte labels such as 'A:min7' "
            f"(qualities {sorted(QUALITY_INTERVALS)}) or `pip install chordsym`"
        ) from None
    tones = chordsym.chord_tones(
        label, dialect="harte" if ":" in label else "default", bass_first=False
    )
    if not tones:
        raise ValueError(f"cannot read chord {label!r}")
    root = tones[0]
    intervals = tuple(sorted({(t - root) % 12 for t in tones}))
    quality = _INTERVALS_TO_QUALITY.get(intervals)
    if quality is None:
        raise ValueError(
            f"chord {label!r} has tones {intervals} above its root, which are not one of "
            f"the supported qualities {sorted(QUALITY_INTERVALS)}"
        )
    return chord_label(root, quality), root, intervals


def default_vocabulary(qualities: Iterable[str] = DFLT_QUALITIES) -> List[str]:
    """Every root x quality as Harte labels, qualities outermost (``C:maj ... B:maj, C:min ...``)."""
    return [chord_label(r, q) for q in qualities for r in range(12)]


def chord_templates(
    vocabulary: Sequence[str], *, harmonics: float = 0.6, n_harmonics: int = 6
) -> Tuple[np.ndarray, List[str]]:
    """``(K, 12)`` unit-norm templates and the Harte labels, one per vocabulary entry.

    ``harmonics`` > 0 adds each chord tone's overtones (harmonic ``h`` weighted ``harmonics**(h-1)``),
    which is how a played chord actually fills the chroma; 0 gives binary templates.
    """
    templates, labels = [], []
    for entry in vocabulary:
        label, root, intervals = parse_chord_label(entry)
        t = np.zeros(12)
        for iv in intervals:
            if harmonics > 0:
                for h in range(n_harmonics):
                    t[(root + iv + _HARMONIC_SEMITONES[h]) % 12] += harmonics**h
            else:
                t[(root + iv) % 12] = 1.0
        templates.append(t / np.linalg.norm(t))
        labels.append(label)
    return np.array(templates), labels


# --- front end -------------------------------------------------------------------------------


def _stft_magnitude(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Hann-windowed magnitudes ``(frames, bins)``; frame ``t`` is centred on sample ``t * hop``."""
    y = np.pad(np.asarray(y, dtype=np.float32), (n_fft // 2, n_fft // 2))
    n = 1 + max(0, (len(y) - n_fft) // hop)
    window = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(y[idx] * window, axis=1)).astype(np.float32)


def peaks_chroma(
    y: np.ndarray,
    sr: int,
    *,
    window_s: float = 0.186,
    hop_s: float = 0.046,
    min_hz: float = 55.0,
    max_hz: float = 2100.0,
    floor_db: float = 50.0,
    log_compression: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(chroma (frames, 12), level_db (frames,))`` from spectral peaks only (HPCP-like).

    Local maxima within ``floor_db`` of the frame's top and between ``min_hz`` and ``max_hz``;
    frequency refined by parabolic interpolation; each peak goes to its nearest pitch class,
    weighted by ``cos^2`` of its distance in semitones, and its magnitude is compressed as
    ``log(1 + log_compression * |X| / frame max)``. Frame ``t`` is centred at ``t * hop_s``.
    """
    n_fft = 1 << int(np.ceil(np.log2(window_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    S = _stft_magnitude(y, n_fft, hop)
    a, b, c = S[:, :-2], S[:, 1:-1], S[:, 2:]
    top = S.max(axis=1, keepdims=True)
    peak = (b > a) & (b >= c) & (b > top * 10 ** (-floor_db / 20))
    la, lb, lc = (np.log(np.maximum(x, 1e-12)) for x in (a, b, c))
    den = la - 2 * lb + lc
    ok = np.abs(den) > 1e-9
    delta = np.clip(
        np.where(ok, 0.5 * (la - lc) / np.where(ok, den, 1.0), 0.0), -0.5, 0.5
    )
    hz = (np.arange(1, S.shape[1] - 1)[None, :] + delta) * sr / n_fft
    keep = peak & (hz >= min_hz) & (hz <= max_hz)
    midi = 69 + 12 * np.log2(np.maximum(hz, 1e-6) / 440.0)
    near = np.round(midi)
    weight = np.cos(np.pi * (midi - near)) ** 2
    value = np.where(
        keep, np.log1p(log_compression * b / np.maximum(top, 1e-12)) * weight, 0.0
    )
    pc = near.astype(int) % 12
    chroma = np.stack([(value * (pc == p)).sum(axis=1) for p in range(12)], axis=1)
    level_db = 10 * np.log10(np.maximum((S.astype(np.float64) ** 2).sum(axis=1), 1e-20))
    return chroma, level_db


# --- decoding ----------------------------------------------------------------------------------


def _cosines(chroma: np.ndarray, templates: np.ndarray) -> np.ndarray:
    unit = chroma / np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-12)
    return unit @ templates.T


def emissions(
    chroma: np.ndarray,
    templates: np.ndarray,
    level_db: Optional[np.ndarray] = None,
    *,
    kappa: float = 20.0,
    gate_db: float = 45.0,
    no_chord_cos: Optional[float] = DFLT_NO_CHORD_COS,
) -> np.ndarray:
    """``(frames, K + 1)`` normalised likelihoods ``exp(kappa * cosine)``; the last column is the
    no-chord state: a fixed cosine ``no_chord_cos`` (a frame is a chord only when some template
    matches better; ``None`` uses the cosine with flat chroma instead, which log compression
    makes far too high), and it wins in frames more than ``gate_db`` below the loudest so far."""
    cos = _cosines(chroma, templates)
    if no_chord_cos is None:
        no = _cosines(chroma, np.ones((1, 12)) / np.sqrt(12))[:, 0]
    else:
        no = np.full(len(chroma), no_chord_cos)
    B = np.exp(kappa * np.column_stack([cos, no]))
    if level_db is not None:
        quiet = level_db < np.maximum.accumulate(level_db) - gate_db
        B[quiet, :-1] *= 1e-6
    return B / B.sum(axis=1, keepdims=True)


def _viterbi(log_B: np.ndarray, log_A: np.ndarray, log_start: np.ndarray) -> np.ndarray:
    T, K = log_B.shape
    d = log_start + log_B[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        m = d[:, None] + log_A
        back[t] = m.argmax(axis=0)
        d = m.max(axis=0) + log_B[t]
    path = np.zeros(T, dtype=int)
    path[-1] = d.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _forward_path(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Causal filter: the argmax of the state posterior given frames up to ``t`` only."""
    p = np.full(B.shape[1], 1.0 / B.shape[1])
    path = np.zeros(len(B), dtype=int)
    for t in range(len(B)):
        p = (p @ A) * B[t]
        p /= p.sum()
        path[t] = p.argmax()
    return path


def _sticky(K: int, stay: float) -> np.ndarray:
    A = np.full((K, K), (1 - stay) / max(K - 1, 1))
    np.fill_diagonal(A, stay)
    return A


def decode_chords(
    chroma: np.ndarray,
    level_db: Optional[np.ndarray] = None,
    *,
    vocabulary: Optional[Sequence[str]] = None,
    hop_s: float = 0.046,
    dwell_s: float = 2.0,
    kappa: float = 20.0,
    gate_db: float = 45.0,
    no_chord_cos: Optional[float] = DFLT_NO_CHORD_COS,
    harmonics: float = 0.6,
    causal: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """``(path, states)``: the state index per frame and the Harte labels (``N`` last).

    ``vocabulary`` restricts the states to those chords (default: every major and minor triad).
    A chord stays for ``dwell_s`` seconds on average. ``causal=True`` reads a frame's decision
    from the frames up to it only (what a live follower can know); default is offline Viterbi.
    """
    templates, labels = chord_templates(
        vocabulary if vocabulary is not None else default_vocabulary(),
        harmonics=harmonics,
    )
    states = labels + [NO_CHORD]
    B = emissions(
        chroma,
        templates,
        level_db,
        kappa=kappa,
        gate_db=gate_db,
        no_chord_cos=no_chord_cos,
    )
    A = _sticky(len(states), 1 - hop_s / dwell_s)
    if causal:
        return _forward_path(B, A), states
    K = len(states)
    path = _viterbi(np.log(B + 1e-300), np.log(A), np.full(K, -np.log(K)))
    return path, states


def align_sequence(
    chroma: np.ndarray,
    sequence: Sequence[str],
    level_db: Optional[np.ndarray] = None,
    *,
    hop_s: float = 0.046,
    kappa: float = 20.0,
    gate_db: float = 45.0,
    no_chord_cos: Optional[float] = DFLT_NO_CHORD_COS,
    harmonics: float = 0.6,
    stay: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Force the decode through ``sequence`` in order (repeats allowed): a left-to-right HMM.

    Returns ``(path, labels, positions)``: ``path[t]`` indexes ``labels``, one label per
    position of ``[N, *sequence, N]`` (a no-chord state may precede and follow the sequence, for
    an intro or an outro); ``positions[t]`` is the sequence index of frame ``t`` (``-1`` for the
    no-chord ends), which keeps two equal neighbouring chords apart.
    """
    if not len(sequence):
        raise ValueError("sequence is empty")
    n = len(sequence)
    if len(chroma) < n:
        raise ValueError(
            f"audio has {len(chroma)} frames, fewer than the {n} chords of the sequence"
        )
    parsed = [parse_chord_label(c)[0] for c in sequence]
    unique = list(dict.fromkeys(parsed))
    templates, unique_labels = chord_templates(unique, harmonics=harmonics)
    B = emissions(
        chroma,
        templates,
        level_db,
        kappa=kappa,
        gate_db=gate_db,
        no_chord_cos=no_chord_cos,
    )
    col = {lab: i for i, lab in enumerate(unique_labels)}
    n_col = B.shape[1] - 1  # the no-chord column
    state_col = [n_col] + [col[p] for p in parsed] + [n_col]
    labels = [NO_CHORD] + parsed + [NO_CHORD]
    S = n + 2
    if stay is None:
        stay = max(
            0.5, 1 - n / len(chroma)
        )  # expected dwell: the frames per chord of the sequence
    log_A = np.full((S, S), -np.inf)
    for i in range(S):
        log_A[i, i] = np.log(stay)
        if i + 1 < S:
            log_A[i, i + 1] = np.log(1 - stay)
    log_A[S - 1, S - 1] = 0.0  # the trailing no-chord state is absorbing
    log_B = np.log(B[:, state_col] + 1e-300)
    start = np.full(S, -np.inf)
    start[0] = start[1] = 0.0
    d = start + log_B[0]
    T = len(B)
    back = np.zeros((T, S), dtype=int)
    for t in range(1, T):
        m = d[:, None] + log_A
        back[t] = m.argmax(axis=0)
        d = m.max(axis=0) + log_B[t]
    end = np.array([S - 2, S - 1])  # the last chord, or the trailing no-chord state
    last = int(end[np.argmax(d[end])])
    path = np.zeros(T, dtype=int)
    path[-1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    positions = [int(p) - 1 if 1 <= p <= n else -1 for p in path]
    return path, labels, positions


def path_to_segments(
    path: Sequence[int],
    labels: Sequence[str],
    *,
    hop_s: float,
    duration: Optional[float] = None,
    split_on: Optional[Sequence[int]] = None,
) -> List[dict]:
    """Run-length the per-frame path into ``[{start, end, label}]`` (frame ``t`` spans
    ``t * hop_s`` +- half a hop). ``split_on`` (e.g. sequence positions) also cuts where it changes."""
    key = list(split_on) if split_on is not None else list(path)
    segments = []
    s = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or key[t] != key[s]:
            start = max(0.0, (s - 0.5) * hop_s)
            end = (t - 0.5) * hop_s
            if duration is not None:
                end = min(end, duration)
            segments.append(
                {
                    "start": round(start, 4),
                    "end": round(end, 4),
                    "label": labels[path[s]],
                }
            )
            s = t
    if duration is not None and segments:
        segments[-1]["end"] = round(duration, 4)
    return segments
