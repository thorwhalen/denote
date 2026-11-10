"""Configuration for the PESTO pitch backend."""

BACKEND_CONFIG = {
    "name": "pesto",
    "display_name": "PESTO (Sony CSL)",
    "pip_install": "pesto-pitch",
    "import_name": "pesto",
    "license": "LGPL-3.0",
    # The checkpoints (pesto/weights/*.ckpt) ship inside the LGPL-3.0 repository and
    # wheel with no separate licence: https://github.com/SonyCSLParis/pesto/blob/master/LICENSE.md
    # LGPL-3.0 puts no restriction on using the outputs; its obligations apply to
    # redistributing the weights or a modified copy.
    "weights_license": "LGPL-3.0",
    "weights_license_note": (
        "checkpoints ship in the LGPL-3.0 repository and wheel with no separate "
        "licence; redistributing them or a modified copy carries LGPL-3.0's terms"
    ),
    "commercial_use": "yes",
    "tasks": ["pitch"],
    "default_for": [],  # torchcrepe stays the default
    "supports_realtime": True,
    "description": (
        "Self-supervised pitch estimator (ISMIR 2023) on a harmonic CQT, about "
        "130K parameters; streaming capable. The 'mir-1k_g7' checkpoint ships in "
        "the wheel, no download. Monophonic sources."
    ),
    "param_map": {
        "audio": {"native_name": "x"},
        "sr": {"native_name": "sr"},
        # milliseconds between frames (the effective hop is rounded to whole samples)
        "step_size_ms": {"native_name": "step_size", "default": 10.0},
        # a shipped checkpoint name ('mir-1k_g7', 'mir-1k') or a checkpoint path
        "model": {"native_name": "model_name", "default": "mir-1k_g7"},
        # how activations become a pitch: 'alwa' (default), 'argmax' or 'mean'
        "reduction": {"native_name": "reduction", "default": "alwa"},
        "device": {"native_name": "device", "default": "cpu"},
        # peak-normalise the input first: the model's voicing depends on level
        "normalize": {"native_name": "normalize", "default": True},
        # frames whose confidence is below this read NaN
        "confidence_threshold": {"native_name": "confidence_threshold", "default": 0.5},
        # Not supported
        "min_frequency": None,
        "max_frequency": None,
        "hop_length": None,
        "onset_threshold": None,
    },
}
