"""Configuration for the SwiftF0 pitch backend."""

BACKEND_CONFIG = {
    "name": "swift_f0",
    "display_name": "SwiftF0",
    "pip_install": "swift-f0",
    "import_name": "swift_f0",
    "license": "MIT",
    # The ONNX model ships inside the MIT-licensed package:
    # https://github.com/lars76/swift-f0/blob/main/LICENSE
    "weights_license": "MIT",
    "commercial_use": "yes",
    "tasks": ["pitch"],
    "default_for": [],  # torchcrepe stays the default
    "supports_realtime": True,
    "description": (
        "Tiny CNN (a 135 KB ONNX model) for monophonic F0 between "
        "46.875 and 2093.75 Hz, one frame every 16 ms. CPU only, far faster "
        "than CREPE; the model ships in the wheel, no download."
    ),
    "param_map": {
        "audio": {"native_name": "audio"},
        "sr": {"native_name": "sample_rate"},
        "min_frequency": {"native_name": "fmin"},
        "max_frequency": {"native_name": "fmax"},
        # frames whose confidence is below this read NaN (SwiftF0 calibrates 0.5)
        "confidence_threshold": {"native_name": "confidence_threshold", "default": 0.5},
        # peak-normalise the input first: the model's voicing depends on level
        "normalize": {"native_name": "normalize", "default": True},
        # ONNX Runtime threads; None lets it pick the physical core count
        "threads": {"native_name": "threads", "default": None},
        # Not supported
        "model": None,
        "device": None,
        "hop_length": None,
        "onset_threshold": None,
    },
}
