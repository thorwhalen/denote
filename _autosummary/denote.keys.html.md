# denote.keys

Key estimation from pitch-class profiles (the Krumhansl-Schmuckler method).

Correlate a 12-bin pitch-class vector (the mean chroma of a recording, or a note
histogram of a score) with a major and a minor *key profile* rotated to each of the
12 tonics. The best of the 24 Pearson correlations names the key. This module is the
audio-free core; `denote.get_key` computes the chroma and calls it.

```pycon
>>> major = KEY_PROFILES["krumhansl"].major
>>> r = key_from_chroma(major)  # the C major profile itself
>>> r.key, round(r.confidence, 3)
('C major', 1.0)
>>> r.relative
'A minor'
>>> key_from_chroma(np.roll(major, 3)).key  # rotated up three semitones
'Eb major'
>>> parse_key("F# minor")
(6, 'minor')
```

Profiles (`KEY_PROFILES`), each a published table of how well each pitch class
fits a key, tonic first:

- `"krumhansl"`: Krumhansl and Kessler’s probe-tone ratings (Krumhansl & Kessler
  1982; Krumhansl, *Cognitive Foundations of Musical Pitch*, 1990, Table 2.1).
- `"temperley"`: Temperley’s profiles from the Kostka-Payne corpus (Temperley,
  *Music and Probability*, 2007), as listed by Humdrum `keycor`.

Both sets of numbers are as given at [https://extras.humdrum.org/man/keycor/](https://extras.humdrum.org/man/keycor/) (and in
music21’s `analysis.discrete`). Any other `(major, minor)` pair of 12 numbers
can be passed as `profile`.

Key names follow the fewest-accidentals convention (Eb major, C# minor, Gb major,
Eb minor), the one `chordsym` uses for chord spelling.

### Module Attributes

| [`FLAT_RTOL`](#denote.keys.FLAT_RTOL)   | A vector whose spread is below this fraction of its largest value is flat.   |
|--------------------------------------------------------------|------------------------------------------------------------------------------|

### Functions

| [`key_from_chroma`](#denote.keys.key_from_chroma)(chroma, \*[, profile, ...])   | The best of the 24 keys for a chroma vector or chromagram.                    |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`key_name`](#denote.keys.key_name)(tonic_pc, mode)                      | `"Eb major"`: the key on pitch class `tonic_pc` (C = 0) in `mode`.            |
| [`key_scores`](#denote.keys.key_scores)(chroma, \*[, profile])             | `{key name: Pearson correlation}` for all 24 keys, best first.                |
| [`parse_key`](#denote.keys.parse_key)(name)                               | `(tonic_pc, mode)` from a key name.                                           |
| [`pitch_class_vector`](#denote.keys.pitch_class_vector)(chroma, \*[, weights])     | A `(12,)` vector: `chroma` itself, or its mean over frames if `(12, frames)`. |
| [`relative_key`](#denote.keys.relative_key)(name)                            | The key sharing `name`'s key signature: C major <-> A minor.                  |
| [`resolve_profile`](#denote.keys.resolve_profile)(profile)                      | A `KeyProfile` from a name in `KEY_PROFILES`, a `KeyProfile` or a pair.       |

### Classes

| [`KeyProfile`](#denote.keys.KeyProfile)(major, minor[, source])   | A major and a minor key profile: 12 weights each, tonic first (C = index 0).   |
|---------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|

### denote.keys.FLAT_RTOL *= 1e-06*

A vector whose spread is below this fraction of its largest value is flat.

### *class* denote.keys.KeyProfile(major, minor, source='')

Bases: [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)

A major and a minor key profile: 12 weights each, tonic first (C = index 0).

#### major *: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), ...]*

Alias for field number 0

#### minor *: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), ...]*

Alias for field number 1

#### source *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Alias for field number 2

### denote.keys.key_from_chroma(chroma, , profile='krumhansl', weights=None, backend='')

The best of the 24 keys for a chroma vector or chromagram.

* **Parameters:**
  * **chroma** – `(12,)` pitch-class weights (C first) or a `(12, frames)`
    chromagram, averaged over frames.
  * **profile** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`KeyProfile`](#denote.keys.KeyProfile), [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – A name in `KEY_PROFILES` (default `"krumhansl"`), a
    `KeyProfile`, or a `(major, minor)` pair of 12 weights each.
  * **weights** – Optional per-frame weights for the average (e.g. frame RMS).
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Recorded on the result.
* **Return type:**
  [`KeyResult`](denote.base.html.md#denote.base.KeyResult)
* **Returns:**
  KeyResult whose `confidence` is the winner’s correlation, `margin` the
  gap to the runner-up, `scores` every key’s correlation (best first), and
  `raw["chroma"]` the 12-bin vector correlated.

### denote.keys.key_name(tonic_pc, mode)

`"Eb major"`: the key on pitch class `tonic_pc` (C = 0) in `mode`.

```pycon
>>> key_name(3, "major"), key_name(1, "minor")
('Eb major', 'C# minor')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.keys.key_scores(chroma, , profile='krumhansl')

`{key name: Pearson correlation}` for all 24 keys, best first.

Raises `ValueError` when the vector is flat to within `FLAT_RTOL` (silence,
or no pitched content), since a correlation with a constant is undefined.
Level gating (very quiet or very short audio) is the audio adapter’s job:
per-frame chroma normalization makes even faint noise look pitched here.

* **Return type:**
  [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### denote.keys.parse_key(name)

`(tonic_pc, mode)` from a key name.

Accepted: `"Eb major"`, `"c# minor"`, `"A min"`, `"Am"`, `"Amin"`,
`"Amaj"` and a bare uppercase tonic (`"Bb"`, major). A bare *lowercase*
tonic (`"a"`) is rejected, because some conventions read it as minor; say
`"a minor"` or `"am"`.

```pycon
>>> parse_key("Eb major"), parse_key("Am"), parse_key("Bbmaj")
((3, 'major'), (9, 'minor'), (10, 'major'))
```

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### denote.keys.pitch_class_vector(chroma, , weights=None)

A `(12,)` vector: `chroma` itself, or its mean over frames if `(12, frames)`.

`weights` (one non-negative number per frame) makes it a weighted mean, e.g.
by frame energy so quiet frames count less.

* **Return type:**
  `ndarray`

### denote.keys.relative_key(name)

The key sharing `name`’s key signature: C major <-> A minor.

```pycon
>>> relative_key("C major"), relative_key("F# minor")
('A minor', 'A major')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.keys.resolve_profile(profile)

A `KeyProfile` from a name in `KEY_PROFILES`, a `KeyProfile` or a pair.

* **Return type:**
  [`KeyProfile`](#denote.keys.KeyProfile)
