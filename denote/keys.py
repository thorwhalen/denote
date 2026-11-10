"""Key estimation from pitch-class profiles (the Krumhansl-Schmuckler method).

Correlate a 12-bin pitch-class vector (the mean chroma of a recording, or a note
histogram of a score) with a major and a minor *key profile* rotated to each of the
12 tonics. The best of the 24 Pearson correlations names the key. This module is the
audio-free core; ``denote.get_key`` computes the chroma and calls it.

>>> major = KEY_PROFILES["krumhansl"].major
>>> r = key_from_chroma(major)  # the C major profile itself
>>> r.key, round(r.confidence, 3)
('C major', 1.0)
>>> r.relative
'A minor'
>>> key_from_chroma(np.roll(major, 3)).key  # rotated up three semitones
'Eb major'
>>> parse_key("F# minor")
(6, 'minor')

Profiles (``KEY_PROFILES``), each a published table of how well each pitch class
fits a key, tonic first:

- ``"krumhansl"``: Krumhansl and Kessler's probe-tone ratings (Krumhansl & Kessler
  1982; Krumhansl, *Cognitive Foundations of Musical Pitch*, 1990, Table 2.1).
- ``"temperley"``: Temperley's profiles from the Kostka-Payne corpus (Temperley,
  *Music and Probability*, 2007), as listed by Humdrum ``keycor``.

Both sets of numbers are as given at https://extras.humdrum.org/man/keycor/ (and in
music21's ``analysis.discrete``). Any other ``(major, minor)`` pair of 12 numbers
can be passed as ``profile``.

Key names follow the fewest-accidentals convention (Eb major, C# minor, Gb major,
Eb minor), the one ``chordsym`` uses for chord spelling.
"""

from typing import Dict, NamedTuple, Sequence, Tuple, Union

import numpy as np

from denote.base import KeyResult

MODES = ("major", "minor")


class KeyProfile(NamedTuple):
    """A major and a minor key profile: 12 weights each, tonic first (C = index 0)."""

    major: Tuple[float, ...]
    minor: Tuple[float, ...]
    source: str = ""


# fmt: off
KEY_PROFILES: Dict[str, KeyProfile] = {
    "krumhansl": KeyProfile(
        major=(6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        minor=(6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
        source="Krumhansl & Kessler (1982); Krumhansl (1990), Table 2.1",
    ),
    "temperley": KeyProfile(
        major=(0.748, 0.060, 0.488, 0.082, 0.670, 0.460,
               0.096, 0.715, 0.104, 0.366, 0.057, 0.400),
        minor=(0.712, 0.084, 0.474, 0.618, 0.049, 0.460,
               0.105, 0.747, 0.404, 0.067, 0.133, 0.330),
        source="Temperley (2007), Kostka-Payne corpus",
    ),
}
# fmt: on

DEFAULT_PROFILE = "krumhansl"

# Tonic spellings with the fewest accidentals in the key signature (6-accidental ties:
# Gb major, Eb minor), matching chordsym's FLAT_MAJOR_TONICS / FLAT_MINOR_TONICS.
TONIC_NAMES: Dict[str, Tuple[str, ...]] = {
    "major": ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"),
    "minor": ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"),
}

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"#": 1, "♯": 1, "b": -1, "♭": -1}

ProfileSpec = Union[str, KeyProfile, Tuple[Sequence[float], Sequence[float]]]


def key_name(tonic_pc: int, mode: str) -> str:
    """``"Eb major"``: the key on pitch class ``tonic_pc`` (C = 0) in ``mode``.

    >>> key_name(3, "major"), key_name(1, "minor")
    ('Eb major', 'C# minor')
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    return f"{TONIC_NAMES[mode][tonic_pc % 12]} {mode}"


def parse_key(name: str) -> Tuple[int, str]:
    """``(tonic_pc, mode)`` from a name like ``"Eb major"``, ``"c# minor"`` or ``"Am"``.

    >>> parse_key("Eb major"), parse_key("c# minor"), parse_key("Am"), parse_key("Bb")
    ((3, 'major'), (1, 'minor'), (9, 'minor'), (10, 'major'))
    """
    text = name.strip()
    parts = text.split()
    if len(parts) == 2:
        tonic, mode = parts[0], parts[1].lower()
    elif len(parts) == 1 and text.endswith("m") and len(text) > 1:
        tonic, mode = text[:-1], "minor"
    elif len(parts) == 1:
        tonic, mode = text, "major"
    else:
        raise ValueError(f"not a key name: {name!r}")
    mode = {"maj": "major", "min": "minor"}.get(mode, mode)
    letter = tonic[:1].upper()
    if letter not in _LETTER_PC or mode not in MODES:
        raise ValueError(f"not a key name: {name!r}")
    try:
        shift = sum(_ACCIDENTAL[c] for c in tonic[1:])
    except KeyError:
        raise ValueError(f"not a key name: {name!r}") from None
    return (_LETTER_PC[letter] + shift) % 12, mode


def relative_key(name: str) -> str:
    """The key sharing ``name``'s key signature: C major <-> A minor.

    >>> relative_key("C major"), relative_key("F# minor")
    ('A minor', 'A major')
    """
    pc, mode = parse_key(name)
    if mode == "major":
        return key_name(pc - 3, "minor")
    return key_name(pc + 3, "major")


def resolve_profile(profile: ProfileSpec) -> KeyProfile:
    """A ``KeyProfile`` from a name in ``KEY_PROFILES``, a ``KeyProfile`` or a pair."""
    if isinstance(profile, str):
        if profile not in KEY_PROFILES:
            raise ValueError(
                f"unknown key profile {profile!r}; known: {sorted(KEY_PROFILES)}"
            )
        return KEY_PROFILES[profile]
    major, minor = profile[0], profile[1]
    for mode, weights in (("major", major), ("minor", minor)):
        if np.shape(weights) != (12,):
            raise ValueError(f"the {mode} profile needs 12 weights, got {weights!r}")
    source = getattr(profile, "source", "")
    return KeyProfile(tuple(map(float, major)), tuple(map(float, minor)), source)


def _profile_label(profile: ProfileSpec) -> str:
    return profile if isinstance(profile, str) else "custom"


def pitch_class_vector(chroma) -> np.ndarray:
    """A ``(12,)`` vector: ``chroma`` itself, or its mean over frames if ``(12, frames)``."""
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 2 and chroma.shape[0] == 12:
        chroma = chroma.mean(axis=1)
    if chroma.shape != (12,):
        raise ValueError(
            f"chroma must have shape (12,) or (12, frames), got {np.shape(chroma)}"
        )
    return chroma


def key_scores(chroma, *, profile: ProfileSpec = DEFAULT_PROFILE) -> Dict[str, float]:
    """``{key name: Pearson correlation}`` for all 24 keys, best first.

    Raises ``ValueError`` when the vector is flat (silence, or no pitched content),
    since a correlation with a constant is undefined.

    >>> scores = key_scores(KEY_PROFILES["krumhansl"].minor)
    >>> list(scores)[:2], len(scores)
    (['C minor', 'Eb major'], 24)
    """
    vector = pitch_class_vector(chroma)
    if not np.all(np.isfinite(vector)) or np.ptp(vector) == 0:
        raise ValueError(
            "the pitch-class vector is flat (silence or no pitched content): "
            "no key can be estimated"
        )
    p = resolve_profile(profile)
    z = (vector - vector.mean()) / vector.std()
    scores = {}
    for mode, weights in (("major", p.major), ("minor", p.minor)):
        w = np.asarray(weights, dtype=float)
        for tonic in range(12):
            rotated = np.roll(w, tonic)
            zr = (rotated - rotated.mean()) / rotated.std()
            scores[key_name(tonic, mode)] = float(np.mean(z * zr))
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def key_from_chroma(
    chroma, *, profile: ProfileSpec = DEFAULT_PROFILE, backend: str = ""
) -> KeyResult:
    """The best of the 24 keys for a chroma vector or chromagram.

    Args:
        chroma: ``(12,)`` pitch-class weights (C first) or a ``(12, frames)``
            chromagram, averaged over frames.
        profile: A name in ``KEY_PROFILES`` (default ``"krumhansl"``), a
            ``KeyProfile``, or a ``(major, minor)`` pair of 12 weights each.
        backend: Recorded on the result.

    Returns:
        KeyResult whose ``confidence`` is the winner's correlation, ``margin`` the
        gap to the runner-up, ``scores`` every key's correlation (best first), and
        ``raw["chroma"]`` the 12-bin vector correlated.
    """
    vector = pitch_class_vector(chroma)
    scores = key_scores(vector, profile=profile)
    (best, r1), (_, r2) = list(scores.items())[:2]
    tonic, mode = best.split(" ")
    return KeyResult(
        key=best,
        tonic=tonic,
        mode=mode,
        confidence=r1,
        margin=r1 - r2,
        scores=scores,
        profile=_profile_label(profile),
        raw={"chroma": vector},
        backend=backend,
    )
