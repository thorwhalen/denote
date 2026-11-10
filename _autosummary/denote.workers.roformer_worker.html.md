# denote.workers.roformer_worker

RoFormer worker: audio-separator’s RoFormer checkpoints, run in the `roformer` environment.

Loaded standalone by `ps.isolated.Worker` (it imports nothing from denote). The
mix arrives as a `(channels, frames)` float32 `.npy` file at 44.1 kHz; it is
written, next to it, as a 32-bit float WAV (audio-separator reads files), separated,
and each stem read back and written as `<stem>.npy` (lower-case stem name) in the
request’s folder. Output WAVs are float too (audio-separator keeps the input’s
subtype), so no level is lost to 16-bit quantization; audio-separator scales a stem
down only if its peak exceeds `normalization_threshold` (1.0 here, its maximum).

Stems come back target first (the checkpoint’s primary stem, e.g. `vocals`), then
the rest in the order audio-separator wrote them (it writes the secondary stem first).
audio-separator writes its outputs into one folder fixed when it is built: a
temporary folder of this worker, emptied after each request and removed at exit.

### Module Attributes

| [`STEM_IN_FILENAME`](#denote.workers.roformer_worker.STEM_IN_FILENAME)        | audio-separator names outputs `<input>_(<Stem>)_<model>.<ext>`.                   |
|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`MIX_WAV`](#denote.workers.roformer_worker.MIX_WAV)                 | The input's file name in the request's folder (its stem prefixes outputs).        |
| [`SAMPLE_RATE`](#denote.workers.roformer_worker.SAMPLE_RATE)             | The rate audio-separator resamples to and writes at.                              |
| [`NORMALIZATION_THRESHOLD`](#denote.workers.roformer_worker.NORMALIZATION_THRESHOLD) | Peak above which audio-separator scales a stem down (1.0 is its largest allowed). |

### Functions

| [`load`](#denote.workers.roformer_worker.load)(\*, model_file_dir[, log_level])        | The worker's state: audio-separator's settings and its output folder.         |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`separate`](#denote.workers.roformer_worker.separate)(state, \*, mix, out_dir, sr, model) | Separate the mix in `mix` (a `.npy` path); write `<stem>.npy` into `out_dir`. |
| [`stem_name`](#denote.workers.roformer_worker.stem_name)(path)                              | The lower-cased stem name in an audio-separator output file name.             |
| [`target_first`](#denote.workers.roformer_worker.target_first)(names, primary)                 | `names` with `primary` (case-insensitive) moved to the front, if present.     |

### denote.workers.roformer_worker.MIX_WAV *= 'mix.wav'*

The input’s file name in the request’s folder (its stem prefixes outputs).

### denote.workers.roformer_worker.NORMALIZATION_THRESHOLD *= 1.0*

Peak above which audio-separator scales a stem down (1.0 is its largest allowed).

### denote.workers.roformer_worker.SAMPLE_RATE *= 44100*

The rate audio-separator resamples to and writes at.

### denote.workers.roformer_worker.STEM_IN_FILENAME *= re.compile('_\\\\(([^)]+)\\\\)_')*

audio-separator names outputs `<input>_(<Stem>)_<model>.<ext>`.

### denote.workers.roformer_worker.load(, model_file_dir, log_level='WARNING')

The worker’s state: audio-separator’s settings and its output folder.

### denote.workers.roformer_worker.separate(state, , mix, out_dir, sr, model, sources=None)

Separate the mix in `mix` (a `.npy` path); write `<stem>.npy` into `out_dir`.

* **Returns:**
  {name: path}, “sr”: rate, “sources”: the checkpoint’s stems,
  “model”: model}\`\`; only `sources` (default: all, target first) are kept,
  in that order.
* **Return type:**
  ```
  ``
  ```

  {“stems”

### denote.workers.roformer_worker.stem_name(path)

The lower-cased stem name in an audio-separator output file name.

### denote.workers.roformer_worker.target_first(names, primary)

`names` with `primary` (case-insensitive) moved to the front, if present.

```pycon
>>> target_first(["other", "vocals"], "Vocals")
['vocals', 'other']
```
