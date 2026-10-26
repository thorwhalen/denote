# denote.backends.torchcrepe.adapter

Adapter for torchcrepe — GPU-accelerated pitch estimation.

### Module Attributes

| [`CREPE_SR`](#denote.backends.torchcrepe.adapter.CREPE_SR)   | it resamples every input to it.   |
|-------------------------------------------------------------|-----------------------------------|

### Functions

| [`frame_period`](#denote.backends.torchcrepe.adapter.frame_period)(hop_length, sr)   | Seconds between torchcrepe's frames for `hop_length` samples at `sr`.   |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------|

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

### denote.backends.torchcrepe.adapter.CREPE_SR *= 16000*

it resamples every input to it.

* **Type:**
  torchcrepe’s model rate

### denote.backends.torchcrepe.adapter.frame_period(hop_length, sr)

Seconds between torchcrepe’s frames for `hop_length` samples at `sr`.

torchcrepe resamples to 16 kHz and truncates the hop there
(`int(hop * 16000 / sr)`), so the frame period is that over 16000, not
`hop / sr`: at 22050 Hz a 220-sample hop is really 219.12 samples, and
times built from `hop / sr` drift 0.4 % (1.2 s at 5 minutes).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
