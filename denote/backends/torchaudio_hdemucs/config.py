"""Configuration for the torchaudio Hybrid Demucs separation backend."""

BACKEND_CONFIG = {
    "name": "torchaudio_hdemucs",
    "display_name": "Hybrid Demucs (torchaudio)",
    "pip_install": "torch torchaudio",
    "import_name": "torchaudio",
    "license": "BSD-2-Clause (torchaudio); check the pretrained weights' terms",
    "tasks": ["separate"],
    "default_for": ["separate"],
    "supports_realtime": False,
    "description": (
        "Four-stem separation (drums, bass, other, vocals) with torchaudio's "
        "HDEMUCS_HIGH_MUSDB_PLUS bundle, run in overlapping chunks so memory "
        "stays flat on long recordings."
    ),
    "param_map": {
        "audio": {"native_name": "audio"},
        "sr": {"native_name": "sr"},
        "segment_s": {"native_name": "segment_s", "default": 10.0},
        "overlap_s": {"native_name": "overlap_s", "default": 1.0},
        "device": {"native_name": "device", "default": "cpu"},
    },
}
