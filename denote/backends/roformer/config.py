"""Configuration for the RoFormer separation backend (audio-separator)."""

from pathlib import Path

#: Every package of the environment, pinned (``python -m denote.isolated lock roformer``).
ENV_LOCK = Path(__file__).with_name("env.lock")

#: The checkpoint used when none is named: Kimberley Jensen's Mel-Band RoFormer
#: vocals model. audio-separator's ``vocals_mel_band_roformer.ckpt`` is the file
#: published at https://huggingface.co/KimberleyJSN/melbandroformer
#: (``MelBandRoformer.ckpt``, same SHA-256 below), whose model card states
#: ``license: mit``. (In UVR's naming, "Kim Vocal 2" is a different, MDX-Net model.)
DFLT_CHECKPOINT = "vocals_mel_band_roformer.ckpt"

#: Weights licence per checkpoint, for checkpoints whose card states one. Any other
#: checkpoint is reported with ``UNLISTED_CHECKPOINT``.
CHECKPOINTS = {
    DFLT_CHECKPOINT: {
        "weights_license": "MIT",
        "commercial_use": "yes",
        "source": "https://huggingface.co/KimberleyJSN/melbandroformer",
        "sha256": "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e",
    },
}
#: What is known of a community checkpoint whose card states no licence.
UNLISTED_CHECKPOINT = {"weights_license": "unstated", "commercial_use": "unclear"}

BACKEND_CONFIG = {
    "name": "roformer",
    "display_name": "RoFormer separation (audio-separator)",
    "pip_install": "python -m denote.isolated build roformer",
    "license": "MIT (audio-separator)",
    # The default checkpoint's licence (see CHECKPOINTS for the source); other
    # checkpoints are reported per call in result.raw["weights_license"].
    "weights_license": "MIT",
    "weights_license_note": (
        "default checkpoint only (https://huggingface.co/KimberleyJSN/melbandroformer); "
        "its card is metadata-only (license: mit) and the training data is unstated; "
        "community checkpoints are unstated unless their card states a licence"
    ),
    "commercial_use": "yes",
    "tasks": ["separate"],
    "supports_realtime": False,
    "isolated_env": {
        # Top-level requirements; the environment is installed from the lock resolved
        # from them (ENV_LOCK). audio-separator needs torch>=2.13 on Apple silicon;
        # [cpu] adds onnxruntime. 0.48.0 imports audioread without declaring it.
        "requirements": ["audio-separator[cpu]==0.48.0", "audioread>=3.0"],
        "lock": str(ENV_LOCK),
        "python": "3.12",
        # PyPI's torch on Linux carries CUDA libraries: several GB more there
        "size_hint": (
            "1 GB on macOS (several GB on Linux, CUDA torch), plus about 900 MB "
            "for the default checkpoint"
        ),
    },
    "description": (
        "Band-split / Mel-band RoFormer checkpoints through audio-separator, in its "
        "own environment (torch>=2.13). The default checkpoint (Kimberley Jensen's "
        "Mel-Band RoFormer) splits vocals from the rest: stems 'vocals', then "
        "'other' (the checkpoint's target stem always comes first)."
    ),
    "param_map": {
        "audio": {"native_name": "audio"},
        "sr": {"native_name": "sr"},
        # a checkpoint file name audio-separator knows (downloaded on first use)
        "model": {"native_name": "model", "default": DFLT_CHECKPOINT},
        # stem names to keep (lower case), e.g. ("vocals",); None keeps all
        "sources": {"native_name": "sources", "default": None},
        # build the environment if missing (else EnvNotBuilt says how)
        "build_env": {"native_name": "build_env", "default": False},
        "device": None,  # audio-separator picks MPS / CUDA / CPU itself
    },
}
