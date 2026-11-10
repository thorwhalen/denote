# SwiftF0 Backend — AI Context

## What it does
Monophonic F0 with a tiny ONNX CNN, CPU only, one frame every 16 ms, 46.875 to 2093.75 Hz.

## Native API
```python
from swift_f0 import SwiftF0

detector = SwiftF0(threads=None)
r = detector.detect(audio, sample_rate, fmin=None, fmax=None)
r.timestamps, r.pitch_hz, r.confidence, r.loudness_db
```

## Key gotchas
- Confidence is calibrated so a frame is voiced at >= 0.5; the adapter sets frames below `confidence_threshold` (default 0.5) to NaN.
- Trained for harmonic sources: a pure sine reads lower confidence (median about 0.5) than a tone with harmonics.
- The adapter resamples to 16 kHz with librosa, so swift-f0's optional soxr is not needed.
- The model ships in the wheel: no download.

## License
MIT, code and model (https://github.com/lars76/swift-f0).
