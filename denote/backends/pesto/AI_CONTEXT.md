# PESTO Backend — AI Context

## What it does
Self-supervised monophonic pitch estimation (Sony CSL, ISMIR 2023) on a harmonic CQT; about 130K parameters; streaming capable.

## Native API
```python
from pesto import load_model, predict

timesteps_ms, pitch_hz, confidence, activations = predict(
    x, sr, step_size=10.0, model_name="mir-1k_g7"
)
model = load_model(
    "mir-1k_g7", step_size=10.0
)  # reusable; its CQT kernels follow the input's rate
```

## Key gotchas
- PESTO rounds the step to whole samples, so the adapter builds times from `int(step * sr / 1000 + 0.5) / sr`, not the nominal step (see `frame_period`).
- No frequency-range arguments.
- The checkpoints ship in the wheel: no download. Needs torch and torchaudio.

## License
LGPL-3.0, code and the checkpoints shipped in the repository (https://github.com/SonyCSLParis/pesto). Commercial use is allowed; redistributing the weights or a modified copy carries LGPL-3.0's obligations.
