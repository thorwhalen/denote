"""Configuration for the librosa key estimation backend (the single source of its defaults)."""

from denote.keys import DEFAULT_PROFILE
from denote.util import DFLT_AUDIO_SR

#: Audio quieter than this overall RMS (-60 dBFS) is refused as "no music to key".
DFLT_MIN_RMS = 1e-3
#: Audio shorter than this (seconds) is refused: too few notes to imply a key.
DFLT_MIN_DURATION_S = 1.0

BACKEND_CONFIG = {
    "name": "librosa_key",
    "display_name": "librosa chroma + key profiles",
    "pip_install": "librosa>=0.10",
    "import_name": "librosa",
    "license": "ISC",
    # No learned weights: published key profiles (Krumhansl-Kessler, Temperley)
    # over librosa chroma (https://github.com/librosa/librosa/blob/main/LICENSE.md)
    "weights_license": "none (no learned weights)",
    "commercial_use": "yes",
    "tasks": ["key"],
    "default_for": ["key"],
    "supports_realtime": False,
    "description": (
        "Krumhansl-Schmuckler key finding: the recording's energy-weighted mean "
        "chroma (CQT by default) correlated with 24 rotated major/minor key "
        "profiles (Krumhansl-Kessler by default, or Temperley). One key for the "
        "whole input: no modulation tracking. CPU, no model download."
    ),
    "param_map": {
        "audio": {"native_name": "y"},
        # a name in denote.keys.KEY_PROFILES, a KeyProfile, or a (major, minor) pair
        "profile": {"native_name": "profile", "default": DEFAULT_PROFILE},
        # a name in CHROMA_FUNCTIONS ('cqt', 'stft', 'cens') or a callable
        # (y, sr) -> (12, frames) chromagram
        "chroma": {"native_name": "chroma", "default": "cqt"},
        # audio is resampled to this rate first (CQT chroma fails below ~8.2 kHz)
        "audio_sr": {"native_name": "audio_sr", "default": DFLT_AUDIO_SR},
        # how frames are averaged: 'energy' (by frame RMS) or 'uniform'
        "weighting": {"native_name": "weighting", "default": "energy"},
        "min_rms": {"native_name": "min_rms", "default": DFLT_MIN_RMS},
        "min_duration_s": {
            "native_name": "min_duration_s",
            "default": DFLT_MIN_DURATION_S,
        },
        # Not supported
        "model": None,
        "device": None,
    },
}
