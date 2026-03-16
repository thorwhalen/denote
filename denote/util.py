"""Audio loading and utility functions for denote."""

from typing import Union, Optional, Tuple
from pathlib import Path

import numpy as np


def load_audio(
    audio: Union[str, Path, np.ndarray],
    *,
    sr: Optional[int] = None,
    mono: bool = True,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from a file path or validate an existing array.

    Args:
        audio: File path (str/Path) or numpy array of audio samples.
        sr: Sample rate. Required when audio is an array. Ignored for file paths
            (detected automatically).
        mono: If True, convert to mono.
        target_sr: If set, resample to this rate.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        ValueError: If audio is an array and sr is not provided.
        FileNotFoundError: If audio is a path that doesn't exist.
    """
    import librosa

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        y, file_sr = librosa.load(str(path), sr=target_sr, mono=mono)
        return y, target_sr or file_sr

    if isinstance(audio, np.ndarray):
        if sr is None:
            raise ValueError(
                "Sample rate (sr) is required when audio is a numpy array."
            )
        y = audio
        if mono and y.ndim > 1:
            y = librosa.to_mono(y)
        if target_sr is not None and target_sr != sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            return y, target_sr
        return y, sr

    # Try torch.Tensor
    try:
        y = audio.numpy()
        if y.ndim > 1:
            y = y.squeeze()
        if sr is None:
            raise ValueError(
                "Sample rate (sr) is required when audio is a tensor."
            )
        return load_audio(y, sr=sr, mono=mono, target_sr=target_sr)
    except AttributeError:
        pass

    raise TypeError(
        f"Unsupported audio type: {type(audio)}. "
        "Expected str, Path, np.ndarray, or torch.Tensor."
    )


def ensure_file_path(
    audio: Union[str, Path, np.ndarray],
    *,
    sr: Optional[int] = None,
) -> str:
    """Ensure audio is available as a file path.

    If audio is already a path, return it as a string.
    If it's an array, write to a temporary WAV file and return the path.
    """
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return str(path)

    # It's an array — write to temp file
    import tempfile
    import soundfile as sf

    if sr is None:
        raise ValueError(
            "Sample rate (sr) is required when audio is an array."
        )
    y, actual_sr = load_audio(audio, sr=sr, mono=True)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, y, actual_sr)
    return tmp.name
