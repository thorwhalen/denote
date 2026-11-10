# demucs-mlx Backend — AI Context

## What it does
Hybrid Transformer Demucs (v4) source separation on Apple's MLX, via the `demucs-mlx` package: `htdemucs` (default, 4 stems), `htdemucs_ft` (one fine-tuned sub-model per stem, about 4x slower) and `htdemucs_6s` (adds guitar and piano). Apple silicon only.

## How it runs
`demucs-mlx` pins `mlx>=0.32.3,<0.33`, so it runs in its own environment (`denote.isolated`, on `ps.isolated`): `denote/workers/demucs_mlx_worker.py` is loaded by a `ps.isolated.Worker` with that environment's Python and kept alive, so the model loads once. The host resamples to 44.1 kHz and duplicates mono to stereo; audio crosses as `.npy` files in a temporary folder.

```bash
python -m denote.isolated build demucs_mlx   # one-time, about 200 MB
```

```python
stems = denote.separate("song.wav", backend="demucs_mlx").stems
vocals = denote.separate(
    "song.wav", backend="demucs_mlx", model="htdemucs_ft", sources=("vocals",)
)
```

## Native API (inside the environment)
```python
from demucs_mlx import Separator

sep = Separator(model="htdemucs", stem=None)  # stem= only with htdemucs_ft
sep.update_parameter(shifts=1, overlap=0.25)
_, stems = sep.separate_tensor(
    wav
)  # (channels, frames) at 44.1 kHz -> {name: (2, frames)}
```

## Key gotchas
- A missing (or stale) environment raises `denote.isolated.EnvNotBuilt` with the build command; `build_env=True` builds it instead. `$DENOTE_DATA_DIR` moves `envs/`.
- Weights download on first use into `~/.cache/demucs-mlx` (or `$DEMUCS_MLX_CACHE_DIR`), checked against digests in the package. The download uses urllib: the worker points `SSL_CERT_FILE` at certifi's bundle (installed in the environment) unless it is set, because the environment's Python may have no CA bundle of its own.
- demucs-mlx does not normalize its input; the worker does as Demucs's CLI does (minus the mono reference's mean, over its std) and undoes it, adding `mean / n_sources` back to each stem so the stems sum to the mix.
- `sources=("vocals",)` with `htdemucs_ft` runs only the vocals sub-model.
- `model` and `sources` are checked on the host, before any environment or weights are touched.
- The environment is installed from `env.lock` (every package pinned, mlx 0.32.4).
- The worker keeps the model loaded (about 6 GB peak footprint with htdemucs, MLX unified memory) until `denote.services["demucs_mlx"].close()` or exit.

## License
Code MIT (demucs-mlx, Meta's Demucs). Weights: converted from Meta's official checkpoints (https://huggingface.co/ssmall256/demucs-mlx, whose card says MIT); the Demucs author has said those weights were provided for research use (https://huggingface.co/adefossez/HTDemucs/discussions/1). Recorded as `unstated` / `unclear` until settled.
