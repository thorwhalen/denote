# torchaudio Hybrid Demucs Backend — AI Context

## What it does
Four-stem source separation (drums, bass, other, vocals) with torchaudio's pretrained Hybrid Demucs bundles (default `HDEMUCS_HIGH_MUSDB_PLUS`, trained on MUSDB-HQ plus extra data; `HDEMUCS_HIGH_MUSDB` is MUSDB-HQ only). No `demucs` package needed: torch and torchaudio only.

## Native API
```python
import torchaudio

bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().eval()  # weights download once (~320 MB)
est = model(
    x
)  # x: (batch, 2, frames) float32 at bundle.sample_rate (44100), normalized
# est: (batch, 4, 2, frames) in model.sources order: drums, bass, other, vocals
```

## Key gotchas
- Install with `pip install denote[torchaudio_hdemucs]` (torch + torchaudio).
- `bundle=` is a torchaudio.pipelines name: `HDEMUCS_HIGH_MUSDB_PLUS` (default) or `HDEMUCS_HIGH_MUSDB`. The sample rate comes from the bundle (44.1 kHz for both); models are cached per `(bundle, device)`.
- Input must be stereo at the bundle's rate; the adapter duplicates mono, resamples, and accepts `(frames, channels)` arrays and torch tensors.
- Normalize by the mix's mean and std before the model, undo after (as torchaudio's tutorial does). The mean is added back as `mean / 4` per stem, so the four stems sum to the mix.
- The model rejects inputs of 1..768 and 1025..1280 frames (its `_spec` padding); chunks shorter than `MIN_FRAMES` (1281) are zero-padded and cropped back, so any length works.
- Long inputs: the adapter runs `segment_s` chunks through `denote.util.chunked_apply`, overlap-added with linear crossfades over the actual overlap (at least `overlap_s`, 0 allowed) and divided by the summed weights. Every chunk is `segment_s` long (or the whole input, if shorter); the last one ends at the input's end. `overlap_s` must be shorter than `segment_s`.
- Memory: the model's working memory is bounded by `segment_s`, but the output is not: 8 bytes per frame per stem kept (2 channels of float32), plus the normalized input (8 bytes per frame) and a weight vector (4 bytes per frame), all arithmetic in place. For an hour at 44.1 kHz that is about 7 GB with all four stems, or about 3.2 GB with `sources=("vocals",)`, which only stores the stems named (decoding and resampling the file add their own transient copies).
- The weights' license is not stated with the bundle; check before commercial use.
