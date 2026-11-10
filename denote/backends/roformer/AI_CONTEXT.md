# RoFormer Backend — AI Context

## What it does
Source separation with band-split / Mel-band RoFormer checkpoints through `audio-separator`. The default checkpoint, `vocals_mel_band_roformer.ckpt` (Kimberley Jensen's Mel-Band RoFormer), splits a mix into `vocals` and `other`. Any checkpoint file name audio-separator knows can be passed as `model=`.

## How it runs
`audio-separator` needs `torch>=2.13` on Apple silicon, so it runs in its own environment (`denote.isolated`, on `ps.isolated`): `denote/workers/roformer_worker.py` runs in a `ps.isolated.Worker` with that environment's Python, kept alive so a checkpoint loads once. The host resamples to 44.1 kHz and duplicates mono to stereo; the worker writes a float WAV for audio-separator (it reads files) and returns float32 stems as `.npy` files.

```bash
python -m denote.isolated build roformer   # one-time, about 1 GB (torch)
```

```python
vocals = denote.separate("song.wav", backend="roformer").vocals
```

## Native API (inside the environment)
```python
from audio_separator.separator import Separator

sep = Separator(
    model_file_dir=..., output_dir=..., use_soundfile=True, normalization_threshold=1.0
)
sep.load_model(model_filename="vocals_mel_band_roformer.ckpt")
files = sep.separate(
    "mix.wav"
)  # names like mix_(Vocals)_vocals_mel_band_roformer.wav, in output_dir
```

## Key gotchas
- A missing (or stale) environment raises `denote.isolated.EnvNotBuilt` with the build command; `build_env=True` builds it instead.
- Checkpoints download on first use into `$DENOTE_DATA_DIR/models/roformer` (default `~/.local/share/denote/models/roformer`; audio-separator's own default is a folder under `/tmp`). `$AUDIO_SEPARATOR_MODEL_DIR` wins when set (it must exist).
- Stem names come from the output file names, lower-cased (`Vocals` -> `vocals`). audio-separator writes the secondary stem first; the worker puts the checkpoint's target stem (`model_instance.primary_stem_name`) first, so the default checkpoint gives `vocals`, `other`.
- The environment is installed from `env.lock` (every package pinned, universal markers). On Linux PyPI's torch brings CUDA libraries (several GB).
- The worker keeps the checkpoint loaded (about 4.5 GB peak on Apple silicon) until `denote.services["roformer"].close()` or exit.
- Output WAVs keep the input's float subtype; audio-separator scales a stem down only if its peak exceeds 1.0 (`normalization_threshold`, its maximum).
- audio-separator picks the device (MPS on Apple silicon) itself.

## License
Code MIT (audio-separator). Default checkpoint: MIT, per its card at https://huggingface.co/KimberleyJSN/melbandroformer (`MelBandRoformer.ckpt`, SHA-256 `87201f4d...7559e`, the same file as audio-separator's `vocals_mel_band_roformer.ckpt`). Other community checkpoints are reported per call (`result.raw["weights_license"]`, `["commercial_use"]`) as `unstated` / `unclear` unless listed in `config.CHECKPOINTS` with a card that states a licence.
