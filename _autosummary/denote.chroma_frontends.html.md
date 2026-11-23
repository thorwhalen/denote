# denote.chroma_frontends

Chroma front ends for chord recognition: strategies `(y, sr, *, hop_s, window_s) -> (chroma, level_db)`.

`chroma` is `(frames, 12)` with frame `t` centred at `t * hop_s` seconds, `level_db` the
frame energy in dB (for the no-chord gate). `FRONTENDS` maps a name to its function; add one
with `FRONTENDS["mine"] = f` or pass the callable as `frontend=`.

- `peaks`: spectral-peak chroma, no model (`denote.chords.peaks_chroma`); the default.
- `basic_pitch`: Basic Pitch’s note posteriors (a learned polyphonic note model, Apache-2.0)
  folded to 12 pitch classes. Needs `basic-pitch` (the `basic_pitch` extra).
- madmom’s deep chroma (`DeepChromaProcessor`) is a good fit for full mixes but its weights are
  CC BY-NC-SA: wrap it as a callable of this shape for non-commercial use; it is not a default.

### Module Attributes

| [`BASIC_PITCH_LOWEST_MIDI`](#denote.chroma_frontends.BASIC_PITCH_LOWEST_MIDI)   | Basic Pitch's lowest output pitch (A0) as a MIDI number; its note posteriors cover 88 semitones.                                                                                                                                         |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`BASIC_PITCH_FPS`](#denote.chroma_frontends.BASIC_PITCH_FPS)           | 22050 Hz audio, 256-sample hop.                                                                                                                                                                                                          |
| [`FRONTEND_TUNING`](#denote.chroma_frontends.FRONTEND_TUNING)           | Decoder settings each front end was tuned with (GuitarSet dev takes), used when the caller passes none: Basic Pitch's posteriors are sparse and already note-like, so no overtone templates, a softer match and a lower no-chord cosine. |

### Functions

| [`basic_pitch_chroma`](#denote.chroma_frontends.basic_pitch_chroma)(y, sr, \*, hop_s, window_s)   | Basic Pitch `posterior` ('note' or 'onset' or 'contour') folded to pitch classes, averaged to the `hop_s` grid; `power` compresses it (`< 1`) before the fold.   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`frame_level_db`](#denote.chroma_frontends.frame_level_db)(y, sr, hop_s, window_s)           | Frame energy in dB, frames centred at `t * hop_s`.                                                                                                               |
| [`resolve_frontend`](#denote.chroma_frontends.resolve_frontend)(frontend)                       | A front-end callable from a name in `FRONTENDS` or a callable.                                                                                                   |

### denote.chroma_frontends.BASIC_PITCH_FPS *= 86.1328125*

22050 Hz audio, 256-sample hop.

* **Type:**
  Basic Pitch’s model output frame rate

### denote.chroma_frontends.BASIC_PITCH_LOWEST_MIDI *= 21*

Basic Pitch’s lowest output pitch (A0) as a MIDI number; its note posteriors cover 88 semitones.

### denote.chroma_frontends.FRONTEND_TUNING *: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)]* *= {'basic_pitch': {'harmonics': 0.0, 'kappa': 10.0, 'no_chord_cos': 0.4}}*

Decoder settings each front end was tuned with (GuitarSet dev takes), used when the caller
passes none: Basic Pitch’s posteriors are sparse and already note-like, so no overtone
templates, a softer match and a lower no-chord cosine.

### denote.chroma_frontends.basic_pitch_chroma(y, sr, , hop_s, window_s, posterior='note', power=1.0)

Basic Pitch `posterior` (‘note’ or ‘onset’ or ‘contour’) folded to pitch classes, averaged
to the `hop_s` grid; `power` compresses it (`< 1`) before the fold.

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, `ndarray`]

### denote.chroma_frontends.frame_level_db(y, sr, hop_s, window_s)

Frame energy in dB, frames centred at `t * hop_s`.

* **Return type:**
  `ndarray`

### denote.chroma_frontends.resolve_frontend(frontend)

A front-end callable from a name in `FRONTENDS` or a callable.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
