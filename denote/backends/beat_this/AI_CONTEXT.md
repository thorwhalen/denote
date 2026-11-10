# Beat This! Backend — AI Context

## What it does
Beat and downbeat tracking with the CPJKU "Beat This!" transformer (ISMIR 2024). Offline; peak picking by default, madmom's DBN optionally (`dbn=True`, needs madmom).

## Native API
```python
from beat_this.inference import Audio2Beats

tracker = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False, float16=False)
beats, downbeats = tracker(signal, sr)  # mono float array; seconds
```

## Key gotchas
- The checkpoint ('final0', about 80 MB; 'small0' about 8 MB) downloads on first use into torch's hub cache; the adapter caches the loaded tracker per (checkpoint, device, dbn, float16).
- The adapter resamples to 22050 Hz itself (the model's spectrogram rate).
- It returns no tempo: the adapter derives one from the median inter-beat interval.

## License
Code and published weights MIT (README of https://github.com/CPJKU/beat_this). The README notes some training files are copyrighted or under limited Creative Commons licences.
