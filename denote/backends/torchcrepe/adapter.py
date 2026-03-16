"""Adapter for torchcrepe — GPU-accelerated pitch estimation."""

import numpy as np

from denote.base import PitchResult
from denote.util import load_audio
from denote.translation import make_kwargs_translator


class Adapter:
    """torchcrepe adapter for pitch/F0 estimation."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config['param_map'])

    def get_pitch(self, audio, *, sr=None, **kwargs):
        """Estimate pitch from audio using CREPE.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters (min_frequency, max_frequency,
                model, device, hop_length).

        Returns:
            PitchResult with .times, .frequencies, .confidence fields.
        """
        import torch
        import torchcrepe

        y, actual_sr = load_audio(audio, sr=sr, mono=True)

        # torchcrepe expects (1, T) tensor
        audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        native_kwargs = self._translate(sr=actual_sr, **kwargs)
        # Remove audio from native_kwargs (passed positionally)
        native_kwargs.pop('audio', None)

        # Always request periodicity for confidence
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            return_periodicity=True,
            **native_kwargs,
        )

        pitch_np = pitch.squeeze().numpy()
        periodicity_np = periodicity.squeeze().numpy()

        # Build time axis
        hop = native_kwargs.get('hop_length') or int(actual_sr / 100)
        n_frames = len(pitch_np)
        times = np.arange(n_frames) * hop / actual_sr

        # Replace unvoiced frames (low periodicity) with NaN
        frequencies = pitch_np.copy()
        frequencies[periodicity_np < 0.2] = np.nan

        return PitchResult(
            times=times,
            frequencies=frequencies,
            confidence=periodicity_np,
            raw={'pitch': pitch, 'periodicity': periodicity},
            backend='torchcrepe',
        )
