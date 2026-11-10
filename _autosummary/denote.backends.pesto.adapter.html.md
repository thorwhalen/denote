# denote.backends.pesto.adapter

Adapter for PESTO — self-supervised pitch estimation (Sony CSL).

### Functions

| [`frame_period`](#denote.backends.pesto.adapter.frame_period)(step_size_ms, sr)   | Seconds between PESTO's frames for a `step_size_ms` step at `sr`.   |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------|

### Classes

| [`Adapter`](#denote.backends.pesto.adapter.Adapter)(config)   | PESTO adapter for monophonic pitch/F0 estimation.   |
|--------------------------------------------------------------------|-----------------------------------------------------|

### *class* denote.backends.pesto.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

PESTO adapter for monophonic pitch/F0 estimation.

#### get_pitch(audio, , sr=None, \*\*kwargs)

Estimate pitch from audio with PESTO.

* **Parameters:**
  * **audio** – File path, numpy array or torch tensor.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters: `step_size_ms` (default 10),
    `model` (checkpoint name or path), `reduction`, `device`,
    `confidence_threshold` (default 0.5), `normalize` (default
    True: peak-normalise first, since PESTO’s confidence falls with
    level and no frame is voiced at -30 dBFS; False passes the level
    through).
* **Returns:**
  PitchResult with .times, .frequencies (NaN where unvoiced) and
  .confidence; `.raw` holds the pitch, volume and activations.

### denote.backends.pesto.adapter.frame_period(step_size_ms, sr)

Seconds between PESTO’s frames for a `step_size_ms` step at `sr`.

PESTO rounds the step to whole samples (`int(step * sr / 1000 + 0.5)`), so
the true period is that hop over `sr`, not `step_size_ms / 1000`: at
22050 Hz a 10 ms step is 221 samples (10.023 ms), and times built from the
nominal step drift 0.23 % (0.7 s at 5 minutes).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
