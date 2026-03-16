"""Denote — Portal and facade for audio-to-symbol tools.

Simple usage:

    >>> import denote
    >>> result = denote.transcribe("song.wav")         # audio -> MIDI
    >>> chords = denote.get_chords("song.wav")         # audio -> chord labels
    >>> pitch = denote.get_pitch("vocal.wav")           # audio -> F0
    >>> beats = denote.get_beats("song.wav")            # audio -> beat times

    >>> denote.list_backends()                          # see what's available
    >>> denote.list_backends('pitch')                   # backends for a task

Service-level access:

    >>> denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.3)
    >>> denote.services.torchcrepe.get_pitch("vocal.wav", model='tiny')

Native adapter access:

    >>> denote.services.basic_pitch.adapter  # raw adapter instance
"""

from denote.base import (
    AudioInput,
    NoteEvent,
    TranscriptionResult,
    PitchResult,
    ChordResult,
    BeatResult,
    TASKS,
)
from denote.registry import (
    list_backends,
    get_default_backend,
    register_backend,
    get_config,
)
from denote.services import ServiceCollection

# Lazy singleton for service-level access
services = ServiceCollection()


def transcribe(audio, *, sr=None, backend=None, **kwargs):
    """Transcribe audio to MIDI.

    Args:
        audio: File path (str/Path) or numpy array.
        sr: Sample rate (required when audio is an array).
        backend: Backend name. Defaults to 'basic_pitch' if installed.
        **kwargs: Backend-specific parameters (onset_threshold, min_note_length, etc.)

    Returns:
        TranscriptionResult with .midi, .notes, and .raw fields.
    """
    backend = backend or get_default_backend('transcribe')
    handle = services[backend]
    return handle.transcribe(audio, sr=sr, **kwargs)


def get_pitch(audio, *, sr=None, backend=None, **kwargs):
    """Estimate pitch/F0 from audio.

    Args:
        audio: File path (str/Path) or numpy array.
        sr: Sample rate (required when audio is an array).
        backend: Backend name. Defaults to 'torchcrepe' if installed,
            falls back to 'librosa_pyin'.
        **kwargs: Backend-specific parameters (min_frequency, max_frequency,
            model, device, hop_length).

    Returns:
        PitchResult with .times, .frequencies, .confidence fields.
    """
    backend = backend or get_default_backend('pitch')
    handle = services[backend]
    return handle.get_pitch(audio, sr=sr, **kwargs)


def get_chords(audio, *, sr=None, backend=None, **kwargs):
    """Recognize chords from audio.

    Args:
        audio: File path (str/Path) or numpy array.
        sr: Sample rate (required when audio is an array).
        backend: Backend name.
        **kwargs: Backend-specific parameters.

    Returns:
        ChordResult with .intervals, .labels fields.
    """
    backend = backend or get_default_backend('chords')
    handle = services[backend]
    return handle.get_chords(audio, sr=sr, **kwargs)


def get_beats(audio, *, sr=None, backend=None, **kwargs):
    """Track beats in audio.

    Args:
        audio: File path (str/Path) or numpy array.
        sr: Sample rate (required when audio is an array).
        backend: Backend name. Defaults to 'librosa_beats'.
        **kwargs: Backend-specific parameters (hop_length, start_bpm).

    Returns:
        BeatResult with .beats, .downbeats, .tempo fields.
    """
    backend = backend or get_default_backend('beats')
    handle = services[backend]
    return handle.get_beats(audio, sr=sr, **kwargs)
