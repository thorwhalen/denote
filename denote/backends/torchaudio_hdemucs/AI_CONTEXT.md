# torchaudio Hybrid Demucs Backend — AI Context

## What it does
Four-stem source separation (drums, bass, other, vocals) with torchaudio's pretrained `HDEMUCS_HIGH_MUSDB_PLUS` bundle (Hybrid Demucs, trained on MUSDB-HQ plus extra data). No `demucs` package needed: torch and torchaudio only.

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
- Input must be stereo at 44.1 kHz; the adapter duplicates mono and resamples.
- Normalize by the mix's mean and std before the model, undo after (as torchaudio's tutorial does).
- Long inputs run out of memory: the adapter runs `segment_s` chunks overlap-added with linear crossfades of `overlap_s` (`overlap_add_weights`, weights sum to one). `segment_s` must exceed `overlap_s` in samples or the loop never advances; the adapter raises.
- The weights' license is not stated with the bundle; check before commercial use.
