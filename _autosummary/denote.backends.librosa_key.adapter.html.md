# denote.backends.librosa_key.adapter

Adapter for key estimation from librosa chroma and published key profiles.

### Functions

| [`chroma_cens`](#denote.backends.librosa_key.adapter.chroma_cens)(y, sr)                       | CENS chromagram: CQT chroma smoothed and quantized, robust to timbre.       |
|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`chroma_stft`](#denote.backends.librosa_key.adapter.chroma_stft)(y, sr)                       | STFT chromagram: faster than CQT, coarser in the bass.                      |
| [`frame_rms`](#denote.backends.librosa_key.adapter.frame_rms)(y, n_frames, \*[, hop_length]) | RMS per frame, resampled to `n_frames` (so any chroma callable can use it). |

### Classes

| [`Adapter`](#denote.backends.librosa_key.adapter.Adapter)(config)   | Key estimation: weighted mean chroma correlated with 24 rotated key profiles.   |
|--------------------------------------------------------------------|---------------------------------------------------------------------------------|

### *class* denote.backends.librosa_key.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Key estimation: weighted mean chroma correlated with 24 rotated key profiles.

#### get_key(audio, , sr=None, \*\*kwargs)

Estimate the key (tonic + mode) of a whole recording.

* **Parameters:**
  * **audio** – File path or numpy array.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – `profile` (a name in `denote.keys.KEY_PROFILES`, default
    `"krumhansl"`, or a `(major, minor)` pair of 12 weights),
    `chroma` (`"cqt"` default, `"stft"`, `"cens"`, or a callable
    `(y, sr) -> (12, frames)`), `audio_sr` (rate the audio is
    resampled to first, default 22050), `weighting` (`"energy"`
    default: frames weighted by RMS; or `"uniform"`), `min_rms`
    and `min_duration_s` (refuse quieter or shorter audio).
* **Returns:**
  KeyResult; `confidence` is the winning key’s Pearson correlation,
  `margin` its lead over the runner-up, `scores` all 24 keys best
  first, `raw` the averaged chroma, the RMS and the duration.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – audio shorter than `min_duration_s` or quieter than
      `min_rms`, or an unknown `chroma`/`weighting`/`profile`.

### denote.backends.librosa_key.adapter.chroma_cens(y, sr)

CENS chromagram: CQT chroma smoothed and quantized, robust to timbre.

### denote.backends.librosa_key.adapter.chroma_stft(y, sr)

STFT chromagram: faster than CQT, coarser in the bass.

### denote.backends.librosa_key.adapter.frame_rms(y, n_frames, , hop_length=512)

RMS per frame, resampled to `n_frames` (so any chroma callable can use it).
