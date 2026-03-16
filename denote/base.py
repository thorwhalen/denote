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
    backend: str = ''


@dataclass
class PitchResult:
    """Result of pitch/F0 estimation."""

    times: np.ndarray  # timestamps in seconds
    frequencies: np.ndarray  # Hz (NaN for unvoiced frames)
    confidence: np.ndarray  # confidence/voicing probability [0, 1]
    raw: Any = None
    backend: str = ''


@dataclass
class ChordResult:
    """Result of chord recognition."""

    intervals: np.ndarray  # (N, 2) array of [start_time, end_time]
    labels: List[str] = field(default_factory=list)  # chord symbol strings
    raw: Any = None
    backend: str = ''


@dataclass
class BeatResult:
    """Result of beat/downbeat tracking."""

    beats: np.ndarray  # beat times in seconds
    downbeats: np.ndarray = field(default_factory=lambda: np.array([]))
    tempo: Optional[float] = None  # BPM estimate
    raw: Any = None
    backend: str = ''


# Supported task identifiers
TASKS = {
    'transcribe': 'Audio to MIDI transcription',
    'pitch': 'Pitch / F0 estimation',
    'chords': 'Chord recognition',
    'beats': 'Beat and downbeat tracking',
    'separate': 'Source separation',
}
