# denote.workers.demucs_mlx_worker

demucs-mlx worker: HTDemucs on Apple’s MLX, run in the `demucs_mlx` environment.

Loaded standalone by `ps.isolated.Worker` (it imports nothing from denote). The
mix arrives as a `(channels, frames)` float32 `.npy` file at the model’s rate
(44.1 kHz); each requested stem is written back as `<stem>.npy` in the request’s
folder, in the order requested (the model’s order by default).

Normalization is Demucs’s own (`demucs.separate`): the mix minus the mean of its
mono reference, divided by that reference’s std, and undone on the way out; the
mean comes back as `mean / n_sources` per stem, so the stems add up to the mix.
Weights download on first use into demucs-mlx’s cache (`~/.cache/demucs-mlx`, or
`$DEMUCS_MLX_CACHE_DIR`), verified against digests shipped with the package.

### Module Attributes

| [`NORM_EPS`](#denote.workers.demucs_mlx_worker.NORM_EPS)          | Added to the reference's std, so a silent mix does not divide by zero.              |
|--------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`SINGLE_STEM_MODEL`](#denote.workers.demucs_mlx_worker.SINGLE_STEM_MODEL) | The model that has one specialist sub-model per stem (single-stem runs are faster). |
| [`SAMPLE_RATE`](#denote.workers.demucs_mlx_worker.SAMPLE_RATE)       | The rate every demucs model runs at.                                                |

### Functions

| [`load`](#denote.workers.demucs_mlx_worker.load)(\*\*_)                                  | The worker's state: the separator in use, built on first request.             |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`separate`](#denote.workers.demucs_mlx_worker.separate)(state, \*, mix, out_dir, sr[, ...]) | Separate the mix in `mix` (a `.npy` path); write `<stem>.npy` into `out_dir`. |

### denote.workers.demucs_mlx_worker.NORM_EPS *= 1e-08*

Added to the reference’s std, so a silent mix does not divide by zero.

### denote.workers.demucs_mlx_worker.SAMPLE_RATE *= 44100*

The rate every demucs model runs at.

### denote.workers.demucs_mlx_worker.SINGLE_STEM_MODEL *= 'htdemucs_ft'*

The model that has one specialist sub-model per stem (single-stem runs are faster).

### denote.workers.demucs_mlx_worker.load(\*\*\_)

The worker’s state: the separator in use, built on first request.

demucs-mlx downloads weights with urllib, which verifies TLS against the
interpreter’s OpenSSL defaults; an interpreter built without a CA bundle (some
system and MacPorts Pythons) fails there. certifi’s bundle is used unless
`SSL_CERT_FILE` already names one.

### denote.workers.demucs_mlx_worker.separate(state, , mix, out_dir, sr, model='htdemucs', sources=None, shifts=1, overlap=0.25, segment=None, seed=None)

Separate the mix in `mix` (a `.npy` path); write `<stem>.npy` into `out_dir`.

* **Returns:**
  {name: path}, “sr”: rate, “sources”: the model’s stems,
  “model”: model}\`\`; only `sources` (default: all) are written, in that order.
* **Return type:**
  ```
  ``
  ```

  {“stems”
