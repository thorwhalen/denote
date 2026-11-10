# denote.backends.beat_this.adapter

Adapter for Beat This! — transformer beat and downbeat tracking.

### Module Attributes

| [`BEAT_THIS_SR`](#denote.backends.beat_this.adapter.BEAT_THIS_SR)   | The rate Beat This! computes its spectrogram at; audio is resampled to it here.   |
|-----------------------------------------------------------------|-----------------------------------------------------------------------------------|

### Functions

| [`tempo_from_beats`](#denote.backends.beat_this.adapter.tempo_from_beats)(beats)   | BPM from the median inter-beat interval, or `None` with fewer than 2 beats.   |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### Classes

| [`Adapter`](#denote.backends.beat_this.adapter.Adapter)(config)   | Beat This! adapter: beats and downbeats in seconds.   |
|--------------------------------------------------------------------|-------------------------------------------------------|

### *class* denote.backends.beat_this.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Beat This! adapter: beats and downbeats in seconds.

#### get_beats(audio, , sr=None, \*\*kwargs)

Track beats and downbeats with Beat This!.

* **Parameters:**
  * **audio** – File path, numpy array or torch tensor.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters: `model` (checkpoint name, path or
    URL; default ‘final0’), `device`, `dbn` (madmom DBN
    post-processing), `float16`.
* **Returns:**
  BeatResult with .beats, .downbeats (seconds) and .tempo (BPM from the
  median inter-beat interval; None with fewer than two beats).

### denote.backends.beat_this.adapter.BEAT_THIS_SR *= 22050*

The rate Beat This! computes its spectrogram at; audio is resampled to it here.

### denote.backends.beat_this.adapter.tempo_from_beats(beats)

BPM from the median inter-beat interval, or `None` with fewer than 2 beats.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
