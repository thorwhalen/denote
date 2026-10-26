# denote.backends.torchaudio_hdemucs.adapter

Adapter for torchaudio’s Hybrid Demucs bundles (`HDEMUCS_HIGH_MUSDB_PLUS` by default).

### Module Attributes

| [`MIN_FRAMES`](#denote.backends.torchaudio_hdemucs.adapter.MIN_FRAMES)      | Shortest input HDemucs accepts for sure.                                       |
|------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`NORM_EPS`](#denote.backends.torchaudio_hdemucs.adapter.NORM_EPS)        | Added to the mix's std before normalizing, so silence does not divide by zero. |
| [`HDEMUCS_BUNDLES`](#denote.backends.torchaudio_hdemucs.adapter.HDEMUCS_BUNDLES) | The torchaudio.pipelines bundles this adapter runs (four-stem Hybrid Demucs).  |

### Classes

| [`Adapter`](#denote.backends.torchaudio_hdemucs.adapter.Adapter)(config)   | Hybrid Demucs: drums, bass, other, vocals from a stereo mix.   |
|--------------------------------------------------------------------|----------------------------------------------------------------|

### *class* denote.backends.torchaudio_hdemucs.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Hybrid Demucs: drums, bass, other, vocals from a stereo mix.

#### separate(audio, , sr=None, \*\*kwargs)

Separate a mix into stems.

* **Parameters:**
  * **audio** – File path, numpy array or torch tensor (mono, `(frames, channels)`
    or `(channels, frames)`); mono is duplicated to stereo.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters (see `config.param_map`):
    `segment_s` (chunk length; memory per chunk grows with it),
    `overlap_s` (crossfade between chunks, 0 allowed, shorter than
    `segment_s`), `device`, `bundle` (a torchaudio.pipelines name:
    `HDEMUCS_HIGH_MUSDB_PLUS` or `HDEMUCS_HIGH_MUSDB`), `sources`
    (stem names to keep, default all; the others are never stored).
* **Return type:**
  [`SeparationResult`](denote.base.html.md#denote.base.SeparationResult)
* **Returns:**
  SeparationResult with `(2, frames)` stems at the bundle’s rate (44.1 kHz).
  The model’s four stems add up to the mix up to its own error (a few %);
  `.vocals` is mono.

### denote.backends.torchaudio_hdemucs.adapter.HDEMUCS_BUNDLES *= ('HDEMUCS_HIGH_MUSDB_PLUS', 'HDEMUCS_HIGH_MUSDB')*

The torchaudio.pipelines bundles this adapter runs (four-stem Hybrid Demucs).

### denote.backends.torchaudio_hdemucs.adapter.MIN_FRAMES *= 1281*

Shortest input HDemucs accepts for sure. Measured on the torchaudio bundles
(nfft 4096, hop 1024): lengths 769..1024 and >= 1281 run, 1..768 and 1025..1280
fail inside `_spec` (“Spectrogram’s last dimension must be …”), because its
reflect padding falls back to extra zero padding that changes the frame count.
Shorter chunks are zero-padded to this and cropped back.

### denote.backends.torchaudio_hdemucs.adapter.NORM_EPS *= 1e-08*

Added to the mix’s std before normalizing, so silence does not divide by zero.
