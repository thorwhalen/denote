"""Configuration for the librosa key estimation backend."""

BACKEND_CONFIG = {
    "name": "librosa_key",
    "display_name": "librosa chroma + key profiles",
    "pip_install": "librosa>=0.10",
    "import_name": "librosa",
    "license": "ISC",
    "tasks": ["key"],
    "default_for": ["key"],
    "supports_realtime": False,
    "description": (
        "Krumhansl-Schmuckler key finding: the recording's mean chroma (CQT by "
        "default) correlated with 24 rotated major/minor key profiles "
        "(Krumhansl-Kessler by default, or Temperley). One key for the whole "
        "input: no modulation tracking. CPU, no model download."
    ),
    "param_map": {
        "audio": {"native_name": "y"},
        "sr": {"native_name": "sr"},
        # a name in denote.keys.KEY_PROFILES, a KeyProfile, or a (major, minor) pair
        "profile": {"native_name": "profile", "default": "krumhansl"},
        # a name in CHROMA_FUNCTIONS ('cqt', 'stft', 'cens') or a callable
        # (y, sr) -> (12, frames) chromagram
        "chroma": {"native_name": "chroma", "default": "cqt"},
        # Not supported
        "model": None,
        "device": None,
    },
}
