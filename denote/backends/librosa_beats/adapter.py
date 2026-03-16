"""Adapter for librosa beat tracking."""

import numpy as np

from denote.base import BeatResult
from denote.util import load_audio
from denote.translation import make_kwargs_translator


class Adapter:
    """librosa beat tracker adapter."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config['param_map'])

    def get_beats(self, audio, *, sr=None, **kwargs):
        """Track beats in audio using librosa.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters (hop_length, start_bpm, tightness).

        Returns:
            BeatResult with .beats, .tempo fields.
        """
        import librosa

        y, actual_sr = load_audio(audio, sr=sr, mono=True)

        native_kwargs = self._translate(sr=actual_sr, **kwargs)
        native_kwargs.pop('y', None)

        tempo, beat_frames = librosa.beat.beat_track(
            y=y, units='frames', **native_kwargs
        )

        hop = native_kwargs.get('hop_length', 512)
        beat_times = librosa.frames_to_time(beat_frames, sr=actual_sr, hop_length=hop)

        # Handle scalar vs array tempo
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        return BeatResult(
            beats=beat_times,
            downbeats=np.array([]),  # librosa doesn't track downbeats
            tempo=tempo_val,
            raw={'tempo': tempo, 'beat_frames': beat_frames},
            backend='librosa_beats',
        )
