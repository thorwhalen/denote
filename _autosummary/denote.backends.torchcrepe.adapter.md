# denote.backends.torchcrepe.adapter

Adapter for torchcrepe — GPU-accelerated pitch estimation.

### Classes

| [`Adapter`](#denote.backends.torchcrepe.adapter.Adapter)(config)   | torchcrepe adapter for pitch/F0 estimation.   |
|--------------------------------------------------------------------|-----------------------------------------------|

### *class* denote.backends.torchcrepe.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

torchcrepe adapter for pitch/F0 estimation.

#### get_pitch(audio, , sr=None, \*\*kwargs)

Estimate pitch from audio using CREPE.

* **Parameters:**
  * **audio** – File path or numpy array.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters (min_frequency, max_frequency,
    model, device, hop_length).
* **Returns:**
  PitchResult with .times, .frequencies, .confidence fields.
