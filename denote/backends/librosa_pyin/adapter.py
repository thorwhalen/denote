"""Adapter for librosa pYIN — probabilistic pitch estimation."""

import numpy as np

from denote.base import PitchResult
from denote.util import load_audio
from denote.translation import make_kwargs_translator


class Adapter:
    """librosa pYIN adapter for monophonic pitch estimation."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config['param_map'])

    def get_pitch(self, audio, *, sr=None, **kwargs):
        """Estimate pitch from audio using pYIN.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters (min_frequency, max_frequency,
                hop_length, frame_length).

        Returns:
            PitchResult with .times, .frequencies, .confidence fields.
        """
        import librosa

        y, actual_sr = load_audio(audio, sr=sr, mono=True)

        native_kwargs = self._translate(sr=actual_sr, **kwargs)
        # Extract y from native_kwargs (passed as array)
        native_kwargs.pop('y', None)

        f0, voiced_flag, voiced_probs = librosa.pyin(y, **native_kwargs)

        # Build time axis
        hop = native_kwargs.get('hop_length') or 512
        n_frames = len(f0)
        times = librosa.frames_to_time(
            np.arange(n_frames), sr=actual_sr, hop_length=hop
        )

        # Replace unvoiced with NaN
        frequencies = f0.copy()
        frequencies[~voiced_flag] = np.nan

        return PitchResult(
            times=times,
            frequencies=frequencies,
            confidence=voiced_probs,
            raw={'f0': f0, 'voiced_flag': voiced_flag, 'voiced_probs': voiced_probs},
            backend='librosa_pyin',
        )
