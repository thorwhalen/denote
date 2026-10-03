# denote.backends.librosa_beats.adapter

Adapter for librosa beat tracking.

### Classes

| [`Adapter`](#denote.backends.librosa_beats.adapter.Adapter)(config)   | librosa beat tracker adapter.   |
|--------------------------------------------------------------------|---------------------------------|

### *class* denote.backends.librosa_beats.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

librosa beat tracker adapter.

#### get_beats(audio, , sr=None, \*\*kwargs)

Track beats in audio using librosa.

* **Parameters:**
  * **audio** – File path or numpy array.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters (hop_length, start_bpm, tightness).
* **Returns:**
  BeatResult with .beats, .tempo fields.
