# denote.base

Core types and result dataclasses for denote.

### Classes

| [`BeatResult`](#denote.base.BeatResult)(beats[, downbeats, tempo, raw, ...])   | Result of beat/downbeat tracking.                                              |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`ChordResult`](#denote.base.ChordResult)(intervals[, labels, raw, backend])    | Result of chord recognition.                                                   |
| [`KeyResult`](#denote.base.KeyResult)(key, tonic, mode, confidence[, ...])    | Result of key estimation: tonic + mode, with every candidate key's score.      |
| [`NoteEvent`](#denote.base.NoteEvent)(start_time, end_time, pitch, velocity)  | A single note event with timing, pitch, and optional pitch bends.              |
| [`PitchResult`](#denote.base.PitchResult)(times, frequencies, confidence)       | Result of pitch/F0 estimation.                                                 |
| [`SeparationResult`](#denote.base.SeparationResult)(stems, sr[, raw, backend])       | Result of source separation: one `(channels, frames)` array per stem, at `sr`. |
| [`TranscriptionResult`](#denote.base.TranscriptionResult)(midi[, notes, raw, backend])  | Result of audio-to-MIDI transcription.                                         |

### *class* denote.base.BeatResult(beats, downbeats=<factory>, tempo=None, raw=None, backend='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of beat/downbeat tracking.

### *class* denote.base.ChordResult(intervals, labels=list(), raw=None, backend='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of chord recognition.

### *class* denote.base.KeyResult(key, tonic, mode, confidence, margin=0.0, scores=dict(), profile='', raw=None, backend='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of key estimation: tonic + mode, with every candidate key’s score.

`confidence` is the winning key’s score: for profile backends (`librosa_key`)
the Pearson correlation, in [-1, 1], between the recording’s mean chroma and that
key’s profile. It measures fit, and is not a probability. `margin` is the
winner’s score minus the runner-up’s; a small margin means two keys fit about
equally well, most often the relative key (C major / A minor share every pitch
class), the parallel key or a key a fifth away. Look at `ranked` before
trusting the winner when the margin is small.

#### *property* ranked *: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [float](https://docs.python.org/3/builtins/functions.html#float)]]*

`[(key, score), ...]` for every candidate key, best first (`scores`’ order).

#### *property* relative *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The relative key of the estimate (C major -> A minor); `""` if none applies.

Only major and minor have a relative key here; a backend reporting another
mode (dorian, …) gets `""`.

#### *property* runner_up *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The second-best key (`""` when only one was scored).

### *class* denote.base.NoteEvent(start_time, end_time, pitch, velocity, pitch_bends=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A single note event with timing, pitch, and optional pitch bends.

### *class* denote.base.PitchResult(times, frequencies, confidence, raw=None, backend='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of pitch/F0 estimation.

### *class* denote.base.SeparationResult(stems, sr, raw=None, backend='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of source separation: one `(channels, frames)` array per stem, at `sr`.

#### mono(name)

Stem `name` averaged to one channel.

* **Return type:**
  `ndarray`

#### *property* vocals *: ndarray*

`separate(song, sr=sr).vocals`.

* **Type:**
  The vocals stem, mono

### *class* denote.base.TranscriptionResult(midi, notes=list(), raw=None, backend='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of audio-to-MIDI transcription.
