"""Core types and result dataclasses for denote."""

from dataclasses import dataclass, field
from typing import Union, Optional, Any, List, Tuple
from pathlib import Path

import numpy as np

# Type alias for audio input accepted by all facade functions
AudioInput = Union[str, Path, np.ndarray]


@dataclass
class NoteEvent:
    """A single note event with timing, pitch, and optional pitch bends."""

    start_time: float  # seconds
    end_time: float  # seconds
    pitch: int  # MIDI note number (0-127)
    velocity: float  # 0.0 - 1.0
    pitch_bends: Optional[List[int]] = None


@dataclass
class TranscriptionResult:
    """Result of audio-to-MIDI transcription."""

    midi: Any  # pretty_midi.PrettyMIDI (typed as Any to avoid hard dependency)
    notes: List[NoteEvent] = field(default_factory=list)
    raw: Any = None  # backend-specific raw output
    backend: str = ""


@dataclass
class PitchResult:
    """Result of pitch/F0 estimation."""

    times: np.ndarray  # timestamps in seconds
    frequencies: np.ndarray  # Hz (NaN for unvoiced frames)
    confidence: np.ndarray  # confidence/voicing probability [0, 1]
    raw: Any = None
    backend: str = ""


@dataclass
class ChordResult:
    """Result of chord recognition."""

    intervals: np.ndarray  # (N, 2) array of [start_time, end_time]
    labels: List[str] = field(default_factory=list)  # chord symbol strings
    raw: Any = None
    backend: str = ""


@dataclass
class BeatResult:
    """Result of beat/downbeat tracking."""

    beats: np.ndarray  # beat times in seconds
    downbeats: np.ndarray = field(default_factory=lambda: np.array([]))
    tempo: Optional[float] = None  # BPM estimate
    raw: Any = None
    backend: str = ""


@dataclass
class SeparationResult:
    """Result of source separation: one ``(channels, frames)`` array per stem, at ``sr``."""

    stems: dict  # stem name -> np.ndarray (channels, frames)
    sr: int
    raw: Any = None
    backend: str = ""

    def mono(self, name: str) -> np.ndarray:
        """Stem ``name`` averaged to one channel."""
        if name not in self.stems:
            raise KeyError(f"no stem {name!r}; stems: {sorted(self.stems)}")
        return np.asarray(self.stems[name]).mean(axis=0)

    @property
    def vocals(self) -> np.ndarray:
        """The vocals stem, mono: ``separate(song, sr=sr).vocals``."""
        return self.mono("vocals")


@dataclass
class KeyResult:
    """Result of key estimation: tonic + mode, with every candidate key's score.

    ``confidence`` is the winning key's score: for profile backends (``librosa_key``)
    the Pearson correlation, in [-1, 1], between the recording's mean chroma and that
    key's profile. It measures fit, and is not a probability. ``margin`` is the
    winner's score minus the runner-up's; a small margin means two keys fit about
    equally well, most often the relative key (C major / A minor share every pitch
    class), the parallel key or a key a fifth away. Look at ``ranked`` before
    trusting the winner when the margin is small.
    """

    key: str  # "C major", "F# minor"
    tonic: str  # "C", "F#"
    mode: str  # "major" or "minor"
    confidence: float  # the winner's score (correlation for profile backends)
    margin: float = 0.0  # winner's score minus runner-up's
    scores: dict = field(default_factory=dict)  # key name -> score, best first
    profile: str = ""  # key profile used, e.g. "krumhansl"
    raw: Any = None
    backend: str = ""

    @property
    def ranked(self) -> List[Tuple[str, float]]:
        """``[(key, score), ...]`` for every candidate key, best first."""
        return sorted(self.scores.items(), key=lambda kv: -kv[1])

    @property
    def runner_up(self) -> str:
        """The second-best key (``""`` when only one was scored)."""
        ranked = self.ranked
        return ranked[1][0] if len(ranked) > 1 else ""

    @property
    def relative(self) -> str:
        """The relative key of the estimate (C major -> A minor), the usual confusion."""
        from denote.keys import relative_key

        return relative_key(self.key)


# Supported task identifiers
TASKS = {
    "transcribe": "Audio to MIDI transcription",
    "pitch": "Pitch / F0 estimation",
    "chords": "Chord recognition",
    "beats": "Beat and downbeat tracking",
    "separate": "Source separation",
    "key": "Key (tonic + mode) estimation",
}
