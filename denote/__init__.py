"""Denote — Portal and facade for audio-to-symbol tools.

Simple usage::

    import denote
    result = denote.transcribe("song.wav")         # audio -> MIDI
    chords = denote.get_chords("song.wav")         # audio -> chord labels
    pitch = denote.get_pitch("vocal.wav")           # audio -> F0
    beats = denote.get_beats("song.wav")            # audio -> beat times
    key = denote.get_key("song.wav")                # audio -> "A minor" + scores
    sync = denote.align_score("song.mid", "song.wav")  # score <-> recording time map

    denote.list_backends()                          # see what's available
    denote.list_backends('pitch')                   # backends for a task
    denote.backend_table()                          # tasks + weights licence per backend

Service-level access::

    denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.3)
    denote.services.torchcrepe.get_pitch("vocal.wav", model='tiny')

Native adapter access::

    denote.services.basic_pitch.adapter  # raw adapter instance
"""

from denote.base import (
    AudioInput,
    NoteEvent,
    TranscriptionResult,
    PitchResult,
    ChordResult,
    BeatResult,
    SeparationResult,
    KeyResult,
    TASKS,
    COMMERCIAL_USE_VALUES,
)
from denote.registry import (
    list_backends,
    get_default_backend,
    register_backend,
    get_config,
)
from denote.services import ServiceCollection
from denote.align import align_score, align_audio, align_chroma, ScoreAlignment
from denote.compare import compare_audio, AudioComparison
from denote.keys import key_from_chroma, KEY_PROFILES

# Lazy singleton for service-level access
services = ServiceCollection()

#: The config fields :func:`backend_table` reports, in column order.
BACKEND_TABLE_FIELDS = ("name", "tasks", "license", "weights_license", "commercial_use")


def backend_table(task=None, *, fields=BACKEND_TABLE_FIELDS) -> list:
    """One row per backend: its tasks, code licence, weights licence and commercial use.

    Reads the backends' configs only, so no adapter (and no optional dependency)
    is loaded. ``commercial_use`` is one of :data:`COMMERCIAL_USE_VALUES`.

    Args:
        task: If set, only backends for this task ('pitch', 'beats', ...).
        fields: The config keys each row carries (a missing key reads ``None``).

    Returns:
        A list of dicts, one per backend (sorted by name), keyed by ``fields``.

    Example::

        free = [r["name"] for r in denote.backend_table("pitch") if r["commercial_use"] == "yes"]
    """
    return [
        {field: get_config(name).get(field) for field in fields}
        for name in list_backends(task)
    ]


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
    backend = backend or get_default_backend("transcribe")
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
    backend = backend or get_default_backend("pitch")
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
    backend = backend or get_default_backend("chords")
    handle = services[backend]
    return handle.get_chords(audio, sr=sr, **kwargs)


def separate(audio, *, sr=None, backend=None, **kwargs):
    """Separate a mix into stems (drums, bass, other, vocals, ...).

    Args:
        audio: File path (str/Path) or numpy array.
        sr: Sample rate (required when audio is an array).
        backend: Backend name. Defaults to 'torchaudio_hdemucs'. 'demucs_mlx' and
            'roformer' run in their own environment (see ``denote.isolated``).
        **kwargs: Backend-specific parameters. All take ``sources`` (stem names
            to keep, in that order). torchaudio_hdemucs: segment_s, overlap_s,
            device, bundle. demucs_mlx: model ('htdemucs', 'htdemucs_ft',
            'htdemucs_6s'), shifts, overlap, seed, build_env. roformer: model (a
            checkpoint file name), build_env. ``build_env=True`` builds a missing
            environment instead of raising ``denote.isolated.EnvNotBuilt``; the
            worker then keeps the model in memory until
            ``denote.services[backend].close()`` or exit.

    Returns:
        SeparationResult with .stems ({name: (channels, frames)}), .sr, and
        .vocals (mono).

    Example::

        voice = denote.separate("song.mp3").vocals
    """
    backend = backend or get_default_backend("separate")
    handle = services[backend]
    return handle.separate(audio, sr=sr, **kwargs)


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
    backend = backend or get_default_backend("beats")
    handle = services[backend]
    return handle.get_beats(audio, sr=sr, **kwargs)


def get_key(audio, *, sr=None, backend=None, **kwargs):
    """Estimate the key (tonic + mode) of a recording.

    Args:
        audio: File path (str/Path) or numpy array.
        sr: Sample rate (required when audio is an array).
        backend: Backend name. Defaults to 'librosa_key'.
        **kwargs: Backend-specific parameters (``profile``: 'krumhansl' default or
            'temperley' or a (major, minor) pair; ``chroma``: 'cqt' default,
            'stft', 'cens' or a callable (y, sr) -> (12, frames)).

    Returns:
        KeyResult with .key ("A minor"), .tonic, .mode, .confidence (the winner's
        profile correlation, not a probability), .margin (lead over the runner-up),
        .scores (all 24 keys, best first) and .relative.

    Example::

        k = denote.get_key("song.wav")
        k.key, k.margin, k.ranked[:2]
    """
    backend = backend or get_default_backend("key")
    handle = services[backend]
    return handle.get_key(audio, sr=sr, **kwargs)
