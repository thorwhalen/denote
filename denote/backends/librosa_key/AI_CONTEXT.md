# librosa key backend — AI Context

## What it does
One key (tonic + mode) for a whole recording, by the Krumhansl-Schmuckler method: average the chromagram over frames, correlate the 12-bin vector with the major and minor key profiles rotated to all 12 tonics, take the best of the 24. The math lives in `denote/keys.py` (`key_from_chroma`, `key_scores`, `KEY_PROFILES`), usable on any chroma or pitch-class histogram; this adapter only loads audio and computes the chroma.

## Use
```python
import denote

k = denote.get_key("song.wav")  # KeyResult
k.key, k.confidence, k.margin  # e.g. 'A minor', 0.83, 0.04
k.ranked[:3]  # e.g. [('A minor', 0.83), ('C major', 0.79), ...]
denote.get_key(y, sr=sr, profile="temperley", chroma="stft")
denote.key_from_chroma(chromagram)  # precomputed (12,) or (12, frames)
```

## Key gotchas
- `confidence` is a Pearson correlation in [-1, 1], not a probability. `margin` (winner minus runner-up) is the ambiguity signal: when it is small the runner-up is a real alternative.
- The relative key (C major / A minor) shares all seven pitch classes, so it is the classic confusion; the parallel key and keys a fifth away come next. `result.relative` names the relative key; `result.ranked` shows the whole ordering.
- One key for the whole input: modulating music gets the key that dominates on average. For local keys, call it on windows.
- Every chroma frame counts equally (librosa normalizes each frame), so long quiet or noisy stretches dilute the estimate.
- Silence (a flat chroma vector) raises `ValueError`.
- Profiles: `krumhansl` (Krumhansl & Kessler 1982 probe-tone ratings; Humdrum's notes say it leans toward the dominant) and `temperley` (Temperley 2007, Kostka-Payne corpus; Humdrum's notes say it leans toward the relative major for minor keys). Numbers as listed by Humdrum `keycor`, identical to music21's `analysis.discrete`.
- Key names use the fewest-accidentals spelling (Eb major, C# minor; Gb major and Eb minor at six accidentals), as chordsym does.
