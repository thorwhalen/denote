"""Adapter for Basic Pitch (Spotify) — audio to MIDI transcription."""

from denote.base import TranscriptionResult, NoteEvent
from denote.util import ensure_file_path
from denote.translation import make_kwargs_translator


class Adapter:
    """Basic Pitch adapter for polyphonic note transcription."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def transcribe(self, audio, *, sr=None, **kwargs):
        """Transcribe audio to MIDI using Basic Pitch.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters (onset_threshold, min_note_length, etc.)

        Returns:
            TranscriptionResult with .midi, .notes, and .raw fields.
        """
        from denote.backends.basic_pitch.compat import import_basic_pitch

        # Works around basic-pitch's stale pins and prefers the ONNX model (issue #2)
        predict, model_path = import_basic_pitch()

        audio_path = ensure_file_path(audio, sr=sr)
        native_kwargs = self._translate(**kwargs)

        # Remove audio_path from native_kwargs if present (it's positional)
        native_kwargs.pop("audio_path", None)

        model_output, midi_data, note_events = predict(
            audio_path, model_path, **native_kwargs
        )

        notes = [
            NoteEvent(
                start_time=start,
                end_time=end,
                pitch=pitch,
                velocity=amplitude,
                pitch_bends=bends,
            )
            for start, end, pitch, amplitude, bends in note_events
        ]

        return TranscriptionResult(
            midi=midi_data,
            notes=notes,
            raw={"model_output": model_output, "note_events": note_events},
            backend="basic_pitch",
        )
