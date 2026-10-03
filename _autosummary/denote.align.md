# denote.align

Align a symbolic score to a recording of the same music (audio-to-score sync).

[`align_score()`](#denote.align.align_score) answers “which moment of the score is playing at time *t*
of this recording?” (and the reverse), for a score and a performance that
differ in tempo, rubato, intro length, even key. It is the bridge you need to
cut a score and a recording at the same musical point, to compare several
scores of a piece against one recording, or to put score positions on a
recording’s timeline.

Method: chroma features on both sides (the score’s piano roll folded to pitch
classes, no synthesis needed; the audio’s constant-Q chroma), cosine distance,
and dynamic time warping (`librosa.sequence.dtw`). By default the recording
is matched as a *subsequence* of the score, so a clip of the first minute finds
its place inside a full-length score. `transpose="auto"` also tries all
twelve transpositions of the score and keeps the best, because arrangements
are often in another key.

The result’s `cost` (mean cosine distance along the path, 0 = identical
pitch content, 1 = unrelated) is comparable across scores aligned to the same
recording, which makes it a ranking of how faithful each score is.

### Module Attributes

| [`DFLT_FRAME_RATE`](#denote.align.DFLT_FRAME_RATE)   | Default analysis rate (frames per second) for both score and audio.     |
|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`DFLT_AUDIO_SR`](#denote.align.DFLT_AUDIO_SR)     | Default sample rate the audio is resampled to before chroma extraction. |

### Functions

| [`align_audio`](#denote.align.align_audio)(audio, reference, \*[, sr, ...])   | Align two recordings of the same music (a cover, a render, a remix).   |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`align_score`](#denote.align.align_score)(score, audio, \*[, sr, ...])       | Align a score to a recording of (part of) the same music.              |
| [`audio_chroma`](#denote.align.audio_chroma)(audio, \*[, sr, frame_rate, ...]) | L2-normalized 12 x frames constant-Q chroma of a recording.            |
| [`score_chroma`](#denote.align.score_chroma)(score, \*[, frame_rate])          | L2-normalized 12 x frames chroma of a score (drums excluded).          |

### Classes

| [`ScoreAlignment`](#denote.align.ScoreAlignment)(score_times, audio_times, cost)   | A monotone time map between a score's timeline and a recording's.   |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|

### denote.align.DFLT_AUDIO_SR *= 22050*

Default sample rate the audio is resampled to before chroma extraction.

### denote.align.DFLT_FRAME_RATE *= 10.0*

Default analysis rate (frames per second) for both score and audio.

### *class* denote.align.ScoreAlignment(score_times, audio_times, cost, transpose=0, raw=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A monotone time map between a score’s timeline and a recording’s.

#### score_times

Score times in seconds, increasing (one per path step).

#### audio_times

The matching recording times in seconds.

#### cost

Mean cosine distance along the path (lower is a closer match).

#### transpose

Semitones the score was shifted to match the recording.

#### raw

Backend detail (the warping path in frames, the frame rate).

#### audio_to_score(t)

Score time(s) sounding at recording time(s) `t`.

#### *property* score_span *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)*

`(first, last)` score times covered by the recording.

#### score_to_audio(t)

Recording time(s) at which score time(s) `t` sound.

### denote.align.align_audio(audio, reference, , sr=None, reference_sr=None, subsequence=False, transpose=0, frame_rate=10.0, audio_sr=22050)

Align two recordings of the same music (a cover, a render, a remix).

The same chroma DTW as [`align_score()`](#denote.align.align_score), with a recording in the
score’s place: `score_times` are times in `reference`. The `cost` is
a measure of how closely `audio` follows `reference`’s harmony and
melody, tolerant of tempo differences and blind to timbre, which makes it
a fidelity score for generated versions of a piece.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – The recording to place on the reference’s timeline.
  * **reference** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – The reference recording.
  * **sr** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Sample rate when `audio` is an array.
  * **reference_sr** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Sample rate when `reference` is an array.
  * **subsequence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Match `audio` anywhere inside `reference` (default
    `False`: both start and end together).
  * **transpose** (`Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Semitones, or `"auto"` to try all twelve.
  * **frame_rate** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Analysis frames per second.
  * **audio_sr** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Rate both are resampled to before analysis.
* **Return type:**
  [`ScoreAlignment`](#denote.align.ScoreAlignment)
* **Returns:**
  A [`ScoreAlignment`](#denote.align.ScoreAlignment) (`score_*` refers to `reference`).

### denote.align.align_score(score, audio, , sr=None, subsequence=True, transpose=0, frame_rate=10.0, audio_sr=22050)

Align a score to a recording of (part of) the same music.

* **Parameters:**
  * **score** – A `pretty_midi.PrettyMIDI` or a `.mid` path (other score
    formats when `audiate` is installed).
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – A recording: a path, or an array with `sr`.
  * **sr** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Sample rate when `audio` is an array.
  * **subsequence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Match the recording anywhere inside the score (default),
    e.g. a one-minute clip against a full score. `False` forces the
    two to start and end together.
  * **transpose** (`Union`[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Semitones to shift the score’s pitch classes, or `"auto"`
    to try all twelve and keep the lowest-cost one.
  * **frame_rate** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Analysis frames per second for both sides.
  * **audio_sr** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Rate the audio is resampled to before analysis.
* **Return type:**
  [`ScoreAlignment`](#denote.align.ScoreAlignment)
* **Returns:**
  A [`ScoreAlignment`](#denote.align.ScoreAlignment).

Example:

```default
a = denote.align_score("theme.mid", "recording_first_minute.wav")
a.audio_to_score(60.0)      # where the score is when the clip ends
a.cost                      # compare scores of the same piece
```

### denote.align.audio_chroma(audio, , sr=None, frame_rate=10.0, audio_sr=22050)

L2-normalized 12 x frames constant-Q chroma of a recording.

* **Return type:**
  `ndarray`

### denote.align.score_chroma(score, , frame_rate=10.0)

L2-normalized 12 x frames chroma of a score (drums excluded).

* **Return type:**
  `ndarray`
