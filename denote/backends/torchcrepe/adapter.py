"""Adapter for torchcrepe — GPU-accelerated pitch estimation."""

import numpy as np

from denote.base import PitchResult
from denote.util import load_audio
from denote.translation import make_kwargs_translator


#: torchcrepe's model rate: it resamples every input to it.
CREPE_SR = 16000


def frame_period(hop_length: int, sr: int) -> float:
    """Seconds between torchcrepe's frames for ``hop_length`` samples at ``sr``.

    torchcrepe resamples to 16 kHz and truncates the hop there
    (``int(hop * 16000 / sr)``), so the frame period is that over 16000, not
    ``hop / sr``: at 22050 Hz a 220-sample hop is really 219.12 samples, and
    times built from ``hop / sr`` drift 0.4 % (1.2 s at 5 minutes).
    """
    if sr == CREPE_SR:
        return hop_length / sr
    return int(hop_length * CREPE_SR / sr) / CREPE_SR


class Adapter:
    """torchcrepe adapter for pitch/F0 estimation."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

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
        native_kwargs.pop("audio", None)

        # Always request periodicity for confidence
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            return_periodicity=True,
            **native_kwargs,
        )

        pitch_np = pitch.squeeze().numpy()
        periodicity_np = periodicity.squeeze().numpy()

        hop = native_kwargs.get("hop_length") or actual_sr // 100
        times = np.arange(len(pitch_np)) * frame_period(hop, actual_sr)

        # Replace unvoiced frames (low periodicity) with NaN
        frequencies = pitch_np.copy()
        frequencies[periodicity_np < 0.2] = np.nan

        return PitchResult(
            times=times,
            frequencies=frequencies,
            confidence=periodicity_np,
            raw={"pitch": pitch, "periodicity": periodicity},
            backend="torchcrepe",
        )
