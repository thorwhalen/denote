# torchcrepe Backend — AI Context

## What it does
GPU-accelerated pitch/F0 estimation using the CREPE CNN architecture (PyTorch).
Supports Viterbi decoding for temporally smooth output.

## Native API
```python
import torchcrepe

pitch = torchcrepe.predict(
    audio,  # torch.Tensor shape (1, T)
    sample_rate,  # int
    hop_length=None,  # defaults to sr/100 (10ms)
    fmin=50.0,  # Hz
    fmax=2006.0,  # Hz
    model="full",  # 'full' (22M params) or 'tiny'
    decoder=torchcrepe.viterbi,
    return_harmonicity=False,
    return_periodicity=False,
    batch_size=None,
    device="cpu",
    pad=True,
)
```

## Key gotchas
- Input must be a `torch.Tensor` of shape `(1, T)` — adapter wraps numpy→torch
- `hop_length` defaults to `sample_rate / 100` (i.e., 10ms frames)
- Returns `torch.Tensor`; needs `.numpy()` for our standardized output
- Periodicity (confidence) must be explicitly requested via `return_periodicity=True`
- Viterbi decoder is the default and works well; `argmax` is faster but noisier

## Install
```bash
pip install torchcrepe>=0.0.20
```

## License
MIT (permissive)
