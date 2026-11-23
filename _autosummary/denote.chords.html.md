# denote.chords

Model-free chord recognition: spectral-peak chroma, chord templates, a hidden Markov model.

The pieces behind the `chroma_hmm` backend, usable on their own:

- `peaks_chroma(y, sr)`: a `(frames, 12)` chroma from spectral peaks, log-compressed
- `chord_templates(vocabulary)`: unit templates with each tone’s overtones folded in
- `decode_chords(chroma, ...)`: free decoding (Viterbi, or the causal forward filter),
  optionally restricted to a `vocabulary`
- `align_sequence(chroma, sequence)`: a left-to-right decode through a known ordered
  chord list (chord-sequence-to-audio alignment: *when* each chord happens)

Labels are Harte syntax (`C:maj`, `A:min7`, `N` for no chord), the vocabulary of
`mir_eval.chord` and the JAMS chord namespace. All times are seconds.

```pycon
>>> chord_label(9, "min")
'A:min'
>>> parse_chord_label("Am7")
('A:min7', 9, (0, 3, 7, 10))
```

### Module Attributes

| [`QUALITY_INTERVALS`](#denote.chords.QUALITY_INTERVALS)   | Harte shorthand -> semitone intervals above the root.   |
|----------------------------------------------------------------------|---------------------------------------------------------|
| [`DFLT_NO_CHORD_COS`](#denote.chords.DFLT_NO_CHORD_COS)   | flat between 0.4 and 0.6).                              |

### Functions

| [`align_sequence`](#denote.chords.align_sequence)(chroma, sequence[, level_db, ...])   | Force the decode through `sequence` in order (repeats allowed): a left-to-right HMM.                                                                                                                                                                                                                                                                             |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`chord_label`](#denote.chords.chord_label)(root, quality)                          | Harte label of a pitch class and a quality shorthand.                                                                                                                                                                                                                                                                                                            |
| [`chord_templates`](#denote.chords.chord_templates)(vocabulary, \*[, harmonics, ...])   | `(K, 12)` unit-norm templates and the Harte labels, one per vocabulary entry.                                                                                                                                                                                                                                                                                    |
| [`decode_chords`](#denote.chords.decode_chords)(chroma[, level_db, ...])              | `(path, states)`: the state index per frame and the Harte labels (`N` last).                                                                                                                                                                                                                                                                                     |
| [`default_vocabulary`](#denote.chords.default_vocabulary)([qualities])                     | Every root x quality as Harte labels, qualities outermost (<br/><br/>```<br/>``<br/>```<br/><br/>C:maj .                                                                                                                                                                                                                                                         |
| [`emissions`](#denote.chords.emissions)(chroma, templates[, level_db, ...])       | `(frames, K + 1)` normalised likelihoods `exp(kappa * cosine)`; the last column is the no-chord state: a fixed cosine `no_chord_cos` (a frame is a chord only when some template matches better; `None` uses the cosine with flat chroma instead, which log compression makes far too high), and it wins in frames more than `gate_db` below the loudest so far. |
| [`parse_chord_label`](#denote.chords.parse_chord_label)(label)                            | `(harte_label, root_pc, intervals)` of a chord given as Harte (`A:min7`) or in common symbols (`F#m`, `Bb`; sevenths and the like need `chordsym`).                                                                                                                                                                                                              |
| [`path_to_segments`](#denote.chords.path_to_segments)(path, labels, \*, hop_s[, ...])    | Run-length the per-frame path into `[{start, end, label}]` (frame `t` spans `t * hop_s` +- half a hop).                                                                                                                                                                                                                                                          |
| [`peaks_chroma`](#denote.chords.peaks_chroma)(y, sr, \*[, window_s, hop_s, ...])     | `(chroma (frames, 12), level_db (frames,))` from spectral peaks only (HPCP-like).                                                                                                                                                                                                                                                                                |

### denote.chords.DFLT_NO_CHORD_COS *= 0.5*

flat between 0.4 and 0.6).

* **Type:**
  Cosine a frame must beat to be a chord (measured on GuitarSet

### denote.chords.QUALITY_INTERVALS *= {'6': (0, 4, 7, 9), '7': (0, 4, 7, 10), 'aug': (0, 4, 8), 'dim': (0, 3, 6), 'dim7': (0, 3, 6, 9), 'hdim7': (0, 3, 6, 10), 'maj': (0, 4, 7), 'maj7': (0, 4, 7, 11), 'min': (0, 3, 7), 'min6': (0, 3, 7, 9), 'min7': (0, 3, 7, 10), 'sus2': (0, 2, 7), 'sus4': (0, 5, 7)}*

Harte shorthand -> semitone intervals above the root. The SSOT for template tones.

### denote.chords.align_sequence(chroma, sequence, level_db=None, , hop_s=0.046, kappa=20.0, gate_db=45.0, no_chord_cos=0.5, harmonics=0.6, stay=None)

Force the decode through `sequence` in order (repeats allowed): a left-to-right HMM.

Returns `(path, labels, positions)`: `path[t]` indexes `labels`, one label per
position of `[N, *sequence, N]` (a no-chord state may precede and follow the sequence, for
an intro or an outro); `positions[t]` is the sequence index of frame `t` (`-1` for the
no-chord ends), which keeps two equal neighbouring chords apart.

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### denote.chords.chord_label(root, quality)

Harte label of a pitch class and a quality shorthand.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.chords.chord_templates(vocabulary, , harmonics=0.6, n_harmonics=6)

`(K, 12)` unit-norm templates and the Harte labels, one per vocabulary entry.

`harmonics` > 0 adds each chord tone’s overtones (harmonic `h` weighted `harmonics**(h-1)`),
which is how a played chord actually fills the chroma; 0 gives binary templates.

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### denote.chords.decode_chords(chroma, level_db=None, , vocabulary=None, hop_s=0.046, dwell_s=2.0, kappa=20.0, gate_db=45.0, no_chord_cos=0.5, harmonics=0.6, causal=False)

`(path, states)`: the state index per frame and the Harte labels (`N` last).

`vocabulary` restricts the states to those chords (default: every major and minor triad).
A chord stays for `dwell_s` seconds on average. `causal=True` reads a frame’s decision
from the frames up to it only (what a live follower can know); default is offline Viterbi.

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### denote.chords.default_vocabulary(qualities=('maj', 'min'))

Every root x quality as Harte labels, qualities outermost (`C:maj ... B:maj, C:min ...`).

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### denote.chords.emissions(chroma, templates, level_db=None, , kappa=20.0, gate_db=45.0, no_chord_cos=0.5)

`(frames, K + 1)` normalised likelihoods `exp(kappa * cosine)`; the last column is the
no-chord state: a fixed cosine `no_chord_cos` (a frame is a chord only when some template
matches better; `None` uses the cosine with flat chroma instead, which log compression
makes far too high), and it wins in frames more than `gate_db` below the loudest so far.

* **Return type:**
  `ndarray`

### denote.chords.parse_chord_label(label)

`(harte_label, root_pc, intervals)` of a chord given as Harte (`A:min7`) or in common
symbols (`F#m`, `Bb`; sevenths and the like need `chordsym`). Raises `ValueError` when unreadable.

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]]

### denote.chords.path_to_segments(path, labels, , hop_s, duration=None, split_on=None)

Run-length the per-frame path into `[{start, end, label}]` (frame `t` spans
`t * hop_s` +- half a hop). `split_on` (e.g. sequence positions) also cuts where it changes.

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### denote.chords.peaks_chroma(y, sr, , window_s=0.186, hop_s=0.046, min_hz=55.0, max_hz=2100.0, floor_db=50.0, log_compression=100.0)

`(chroma (frames, 12), level_db (frames,))` from spectral peaks only (HPCP-like).

Local maxima within `floor_db` of the frame’s top and between `min_hz` and `max_hz`;
frequency refined by parabolic interpolation; each peak goes to its nearest pitch class,
weighted by `cos^2` of its distance in semitones, and its magnitude is compressed as
`log(1 + log_compression * |X| / frame max)`. Frame `t` is centred at `t * hop_s`.

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, `ndarray`]
