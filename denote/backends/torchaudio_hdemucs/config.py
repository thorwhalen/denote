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
        "Hybrid Demucs bundles, run in overlapping chunks so the model's memory "
        "is bounded by the chunk length. The output itself grows with the "
        "recording: 8 bytes per frame per stem kept (2 channels of float32), "
        "about 1.3 GB per stem-hour at 44.1 kHz; pass sources= to keep fewer."
    ),
    "param_map": {
        "audio": {"native_name": "audio"},
        "sr": {"native_name": "sr"},
        "segment_s": {"native_name": "segment_s", "default": 10.0},
        "overlap_s": {"native_name": "overlap_s", "default": 1.0},
        "device": {"native_name": "device", "default": "cpu"},
        # HDEMUCS_HIGH_MUSDB_PLUS or HDEMUCS_HIGH_MUSDB (the MUSDB-only variant)
        "bundle": {"native_name": "bundle", "default": "HDEMUCS_HIGH_MUSDB_PLUS"},
        # stem names to keep, e.g. ("vocals",); None keeps all (the model always computes
        # all four: this saves memory, not time)
        "sources": {"native_name": "sources", "default": None},
    },
}
