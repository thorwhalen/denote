# denote.backends.librosa_pyin.adapter

Adapter for librosa pYIN — probabilistic pitch estimation.

### Classes

| [`Adapter`](#denote.backends.librosa_pyin.adapter.Adapter)(config)   | librosa pYIN adapter for monophonic pitch estimation.   |
|--------------------------------------------------------------------|---------------------------------------------------------|

### *class* denote.backends.librosa_pyin.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

librosa pYIN adapter for monophonic pitch estimation.

#### get_pitch(audio, , sr=None, \*\*kwargs)

Estimate pitch from audio using pYIN.

* **Parameters:**
  * **audio** – File path or numpy array.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters (min_frequency, max_frequency,
    hop_length, frame_length).
* **Returns:**
  PitchResult with .times, .frequencies, .confidence fields.
