"""Chroma front ends for chord recognition: strategies ``(y, sr, *, hop_s, window_s) -> (chroma, level_db)``.

``chroma`` is ``(frames, 12)`` with frame ``t`` centred at ``t * hop_s`` seconds, ``level_db`` the
frame energy in dB (for the no-chord gate). ``FRONTENDS`` maps a name to its function; add one
with ``FRONTENDS["mine"] = f`` or pass the callable as ``frontend=``.

- ``peaks``: spectral-peak chroma, no model (``denote.chords.peaks_chroma``); the default.
- ``basic_pitch``: Basic Pitch's note posteriors (a learned polyphonic note model, Apache-2.0)
  folded to 12 pitch classes. Needs ``basic-pitch`` (the ``basic_pitch`` extra).
- madmom's deep chroma (``DeepChromaProcessor``) is a good fit for full mixes but its weights are
  CC BY-NC-SA: wrap it as a callable of this shape for non-commercial use; it is not a default.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from denote.chords import peaks_chroma

#: Basic Pitch's lowest output pitch (A0) as a MIDI number; its note posteriors cover 88 semitones.
BASIC_PITCH_LOWEST_MIDI = 21
#: Basic Pitch's model output frame rate: 22050 Hz audio, 256-sample hop.
BASIC_PITCH_FPS = 22050 / 256


def frame_level_db(y: np.ndarray, sr: int, hop_s: float, window_s: float) -> np.ndarray:
    """Frame energy in dB, frames centred at ``t * hop_s``."""
    hop, win = max(1, int(round(hop_s * sr))), int(window_s * sr)
    y = np.pad(np.asarray(y, dtype=np.float64), (win // 2, win // 2))
    n = 1 + max(0, (len(y) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(np.maximum((y[idx] ** 2).sum(axis=1), 1e-20))


def _peaks(y, sr, *, hop_s, window_s, **kwargs):
    return peaks_chroma(y, sr, hop_s=hop_s, window_s=window_s, **kwargs)


def basic_pitch_chroma(
    y: np.ndarray,
    sr: int,
    *,
    hop_s: float,
    window_s: float,
    posterior: str = "note",
    power: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Basic Pitch ``posterior`` ('note' or 'onset' or 'contour') folded to pitch classes, averaged
    to the ``hop_s`` grid; ``power`` compresses it (``< 1``) before the fold."""
    import soundfile as sf
    import tempfile
    import os

    from denote.backends.basic_pitch.compat import import_basic_pitch

    predict, model_path = import_basic_pitch()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, y, sr)
        out, _, _ = predict(path, model_path)
    finally:
        os.unlink(path)
    post = np.asarray(out[posterior], dtype=np.float64)
    if posterior == "contour":  # 3 bins per semitone
        post = post.reshape(len(post), -1, 3).max(axis=2)
    post = post**power
    pcs = (np.arange(post.shape[1]) + BASIC_PITCH_LOWEST_MIDI) % 12
    fold = np.stack([post[:, pcs == p].sum(axis=1) for p in range(12)], axis=1)
    level = frame_level_db(y, sr, hop_s, window_s)
    n = len(level)
    t_src = np.arange(len(fold)) / BASIC_PITCH_FPS
    cell = np.minimum(
        np.maximum(np.round(t_src / hop_s).astype(int), 0), n - 1
    )  # nearest grid frame
    chroma = np.zeros((n, 12))
    count = np.zeros(n)
    np.add.at(chroma, cell, fold)
    np.add.at(count, cell, 1)
    chroma /= np.maximum(count, 1)[:, None]
    return chroma, level


FRONTENDS: Dict[str, Callable] = {"peaks": _peaks, "basic_pitch": basic_pitch_chroma}


#: Decoder settings each front end was tuned with (GuitarSet dev takes), used when the caller
#: passes none: Basic Pitch's posteriors are sparse and already note-like, so no overtone
#: templates, a softer match and a lower no-chord cosine.
FRONTEND_TUNING: Dict[str, dict] = {
    "basic_pitch": {"harmonics": 0.0, "kappa": 10.0, "no_chord_cos": 0.4},
}


def resolve_frontend(frontend) -> Callable:
    """A front-end callable from a name in ``FRONTENDS`` or a callable."""
    if callable(frontend):
        return frontend
    if frontend not in FRONTENDS:
        raise ValueError(
            f"unknown frontend {frontend!r}; pass one of {sorted(FRONTENDS)} or a callable"
        )
    return FRONTENDS[frontend]
