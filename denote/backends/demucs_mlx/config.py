"""Configuration for the demucs-mlx (HTDemucs on MLX) separation backend."""

from pathlib import Path

#: Every package of the environment, pinned (``python -m denote.isolated lock demucs_mlx``).
ENV_LOCK = Path(__file__).with_name("env.lock")

_FOUR_STEMS = ("drums", "bass", "other", "vocals")

#: The models this backend runs, with their stems (in the model's order).
MODELS = {
    "htdemucs": _FOUR_STEMS,
    "htdemucs_ft": _FOUR_STEMS,  # one fine-tuned sub-model per stem
    "htdemucs_6s": _FOUR_STEMS + ("guitar", "piano"),
}

BACKEND_CONFIG = {
    "name": "demucs_mlx",
    "display_name": "HTDemucs on MLX (demucs-mlx)",
    "pip_install": "python -m denote.isolated build demucs_mlx",
    "license": "MIT (demucs-mlx)",
    # The MLX weights (https://huggingface.co/ssmall256/demucs-mlx, card: MIT) are
    # converted from Meta's official Demucs checkpoints. The Demucs author's remarks,
    # quoted in https://huggingface.co/adefossez/HTDemucs/discussions/1, say those
    # weights are not covered by the code's MIT licence and were provided for research
    # use. Unclear until settled (thorwhalen/denote#13).
    "weights_license": "unstated",
    "weights_license_note": (
        "converted from Meta's Demucs checkpoints; author: research use only per "
        "https://huggingface.co/adefossez/HTDemucs/discussions/1 (the MLX copy's "
        "card says MIT)"
    ),
    "commercial_use": "unclear",
    "tasks": ["separate"],
    "supports_realtime": False,
    "platforms": ["macOS arm64"],  # MLX
    "isolated_env": {
        # Top-level requirements; the environment is installed from the lock resolved
        # from them (ENV_LOCK). demucs-mlx pins mlx>=0.32.3,<0.33. It downloads its
        # weights with urllib, which needs a CA bundle the environment's Python may
        # lack: certifi's is used when SSL_CERT_FILE is not set.
        "requirements": ["demucs-mlx==1.6.1", "certifi"],
        "lock": str(ENV_LOCK),
        "python": "3.12",
        "size_hint": "200 MB (macOS arm64), plus 105-640 MB of weights per model",
    },
    "description": (
        "Hybrid Transformer Demucs (v4) on Apple's MLX: four stems (htdemucs, "
        "htdemucs_ft) or six (htdemucs_6s, adds guitar and piano). Runs in its own "
        "environment (demucs-mlx pins its MLX); stems come back as float32 arrays."
    ),
    "param_map": {
        "audio": {"native_name": "audio"},
        "sr": {"native_name": "sr"},
        # htdemucs (fast, 160 MB), htdemucs_ft (best, 4x slower, 640 MB) or htdemucs_6s
        "model": {"native_name": "model", "default": "htdemucs"},
        # stem names to keep, e.g. ("vocals",); None keeps all. With htdemucs_ft one
        # stem runs only that stem's sub-model (about 4x faster).
        "sources": {"native_name": "sources", "default": None},
        # random time shifts averaged (Demucs's test-time augmentation; 0 = none)
        "shifts": {"native_name": "shifts", "default": 1},
        # overlap between the model's segments, as a fraction of a segment
        "overlap": {"native_name": "overlap", "default": 0.25},
        "seed": {"native_name": "seed", "default": None},
        # build the environment if missing (else EnvNotBuilt says how)
        "build_env": {"native_name": "build_env", "default": False},
        "device": None,  # MLX runs on the GPU of Apple silicon
    },
    "models": MODELS,
}
