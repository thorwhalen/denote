# denote.base

Core types and result dataclasses for denote.

### Classes

| [`BeatResult`](#denote.base.BeatResult)(beats[, downbeats, tempo, raw, ...])   | Result of beat/downbeat tracking.                                              |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`ChordResult`](#denote.base.ChordResult)(intervals[, labels, raw, backend])    | Result of chord recognition.                                                   |
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
