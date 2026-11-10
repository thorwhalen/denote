"""Configuration for the Beat This! beat and downbeat tracking backend."""

BACKEND_CONFIG = {
    "name": "beat_this",
    "display_name": "Beat This! (CPJKU)",
    "pip_install": "beat_this",
    "import_name": "beat_this",
    "license": "MIT",
    # "The code and the published model weights are released under the MIT license"
    # (https://github.com/CPJKU/beat_this, README). The same README notes that some
    # training files are copyrighted or under limited Creative Commons licences.
    "weights_license": "MIT",
    "commercial_use": "yes",
    "tasks": ["beats"],
    "default_for": [],  # librosa_beats stays the dependency-free default
    "supports_realtime": False,
    "description": (
        "Transformer beat and downbeat tracker (ISMIR 2024), no DBN needed. "
        "Offline; runs on CPU or GPU. The checkpoint (about 80 MB for 'final0') "
        "is downloaded on first use into torch's hub cache."
    ),
    "param_map": {
        "audio": {"native_name": "signal"},
        "sr": {"native_name": "sr"},
        # a Beat This! checkpoint name ('final0', 'small0', ...), a path or a URL
        "model": {"native_name": "checkpoint_path", "default": "final0"},
        "device": {"native_name": "device", "default": "cpu"},
        # madmom's DBN post-processing instead of peak picking (needs madmom)
        "dbn": {"native_name": "dbn", "default": False},
        "float16": {"native_name": "float16", "default": False},
        # Not supported
        "hop_length": None,
        "start_bpm": None,
        "tightness": None,
    },
}
