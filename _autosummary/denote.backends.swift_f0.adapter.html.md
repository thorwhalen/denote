# denote.backends.swift_f0.adapter

Adapter for SwiftF0 — fast CNN pitch estimation on CPU (ONNX).

### Module Attributes

| [`SWIFT_F0_SR`](#denote.backends.swift_f0.adapter.SWIFT_F0_SR)   | SwiftF0's model rate; audio is resampled to it here (so no soxr is needed).   |
|----------------------------------------------------------------|-------------------------------------------------------------------------------|

### Classes

| [`Adapter`](#denote.backends.swift_f0.adapter.Adapter)(config)   | SwiftF0 adapter for monophonic pitch/F0 estimation.   |
|--------------------------------------------------------------------|-------------------------------------------------------|

### *class* denote.backends.swift_f0.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

SwiftF0 adapter for monophonic pitch/F0 estimation.

#### get_pitch(audio, , sr=None, \*\*kwargs)

Estimate pitch from audio with SwiftF0.

* **Parameters:**
  * **audio** – File path, numpy array or torch tensor.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters: `min_frequency`, `max_frequency`
    (clipped to the model’s 46.875-2093.75 Hz), `confidence_threshold`
    (default 0.5, SwiftF0’s calibrated voicing threshold), `threads`,
    `normalize` (default True: peak-normalise first, since SwiftF0
    calls quiet input unvoiced; frames peaking below 1e-3 are forced
    unvoiced. False passes the level through).
* **Returns:**
  PitchResult with .times, .frequencies (NaN where unvoiced) and
  .confidence; `.raw` is SwiftF0’s own result (with loudness_db).

### denote.backends.swift_f0.adapter.SWIFT_F0_SR *= 16000*

SwiftF0’s model rate; audio is resampled to it here (so no soxr is needed).
