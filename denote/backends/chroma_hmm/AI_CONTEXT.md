# chroma_hmm chord backend — AI Context

## What it does
Chords with no model. `denote/chords.py` holds the mechanism: chroma from spectral peaks (log-compressed), unit chord templates with each tone's overtones folded in, an HMM with sticky self-transitions (mean dwell 2 s), a no-chord state. This adapter loads audio (resampled to 22050 Hz) and calls it.

## Use
```python
import denote

r = denote.get_chords("song.wav")  # ChordResult: Harte labels, "N" = no chord
r.segments()  # [{start, end, label}]
denote.get_chords(y, sr=sr, causal=True)  # forward filter: a frame sees only the past
denote.get_chords(a, vocabulary=["C", "G", "Am", "F"])  # restrict the states
denote.get_chords(a, sequence=["C", "G", "Am", "F"])  # known order, find the times
denote.get_chords(a, qualities=("maj", "min", "7"))  # a bigger default vocabulary
```

## Key gotchas
- Labels are Harte (`C:maj`, `A:min7`). `vocabulary`/`sequence` take Harte labels, or symbols such as `Am7` if `chordsym` is installed; output is always Harte.
- `sequence=` is a left-to-right decode: one segment per entry (equal neighbours stay separate; `raw["positions"]` gives the entry index), optional `N` before and after. It excludes `vocabulary=` and `causal=`. It cannot skip an entry: if the score has repeats or jumps the audio does not play, expand the list to what is played.
- `sections=` (one key per `sequence` entry, e.g. a score system) makes the order tolerant, JumpDTW-style: jumps between section starts (`jump_p`, default 0.01; repeats, skips, D.S.), one skipped entry (`skip_p`), free start and end. Use it only when the recording may deviate from the score: on a recording that follows the score it costs accuracy (free jumps latch onto repeated progressions). On GuitarSet takes rearranged to play the second half, the first half, the second half again: strict 62.5 vs sections 68.3 majmin.
- The no-chord cosine is fixed (`no_chord_cos=0.5`). The natural choice, the cosine with flat chroma, is far too high after log compression and labels almost everything `N` (0.46 vs 0.80 majmin on a GuitarSet subset). Re-tune it if you change the front end.
- Full mixes (voices, drums) are weaker than solo guitar: a learned chroma front end belongs behind the same `backend=` seam.
- Larger vocabularies lose majmin accuracy on solo guitar (7ths confuse the triad): keep the vocabulary to what you need.
