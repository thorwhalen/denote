# denote.compare

Compare a version of a piece with a reference: harmony and sound, on the same moments (denote#3).

[`align_audio()`](denote.align.html.md#denote.align.align_audio) measures how closely a version follows the
reference’s harmony and melody (chroma DTW), and is blind to timbre on purpose.
[`compare_audio()`](#denote.compare.compare_audio) adds the other half, how close the *sound* is, and keeps
both on one footing: the two recordings are aligned once, and every metric reads
the same warping path, so each compares the same musical moments.

Metrics are a registry ([`METRICS`](#denote.compare.METRICS), a `name -> metric` mapping you can
extend or replace through `metrics=`): a metric is a callable taking an
[`AlignedPair`](#denote.compare.AlignedPair) and returning a distance (lower is closer). An embedding
metric (CLAP, MERT, a Fréchet-style distance) is one more entry, with no change
to callers.

Built in:

- `"chroma"`: the alignment’s own cost, mean cosine distance between chroma
  frames along the path (0 identical pitch content, 1 unrelated). Measured on an
  orchestral minute, an unrelated piece in the same key and style scored about
  0.16-0.21 against the reference, so it discriminates mainly below about 0.15.
- `"timbre"`: MFCCs `c1..c19` (`c0`, loudness, left out) of both
  recordings, each coefficient z-scored with statistics pooled over both (so a
  global difference in spectral shape is kept, not normalised away), then the
  root-mean-square difference per coefficient, averaged along the path. 0 is the
  same sound; two unrelated timbres are typically around 1. The starter it
  replaces (z-scored per recording) separated a General MIDI render
  (0.46-0.79) from a light audio-to-audio pass of the original (0.41); it is a
  coarse spectral-envelope measure, not a perceptual one.

### Module Attributes

| [`Metric`](#denote.compare.Metric)       | an aligned pair -> a distance (lower is closer).                                                            |
|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [`METRICS`](#denote.compare.METRICS)      | The built-in metrics, by name.                                                                              |
| [`DFLT_METRICS`](#denote.compare.DFLT_METRICS) | What [`compare_audio()`](#denote.compare.compare_audio) computes when `metrics` is not given. |

### Functions

| [`compare_audio`](#denote.compare.compare_audio)(audio, reference, \*[, ...])   | How close `audio` is to `reference`, metric by metric, on the same musical moments.   |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`chroma_metric`](#denote.compare.chroma_metric)(pair)                          | The alignment's chroma cost: mean cosine distance along the path.                     |
| [`timbre_metric`](#denote.compare.timbre_metric)(pair, \*[, n_mfcc])            | Mean RMS difference of pooled-z-scored MFCCs `c1..` along the path; see the module.   |

### Classes

| [`AlignedPair`](#denote.compare.AlignedPair)(audio, reference, sr, ...)   | Two recordings at one rate, and the alignment between them, as metrics see them.    |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`AudioComparison`](#denote.compare.AudioComparison)(scores, alignment)       | Distances between a version and a reference (lower is closer), and their alignment. |

### *class* denote.compare.AlignedPair(audio, reference, sr, frame_rate, alignment)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Two recordings at one rate, and the alignment between them, as metrics see them.

`audio` and `reference` are mono samples at `sr`; `alignment.raw["path"]`
holds the warping path as `(audio frame, reference frame)` rows at
`frame_rate` frames per second (hop `hop_length` samples).

### *class* denote.compare.AudioComparison(scores, alignment)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Distances between a version and a reference (lower is closer), and their alignment.

`scores` maps each metric name to its distance; `comparison["timbre"]`
reads one.

### denote.compare.DFLT_METRICS *= ('chroma', 'timbre')*

What [`compare_audio()`](#denote.compare.compare_audio) computes when `metrics` is not given.

### denote.compare.METRICS *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[AlignedPair](#denote.compare.AlignedPair)], [float](https://docs.python.org/3/builtins/functions.html#float)]]* *= mappingproxy({'chroma': <function chroma_metric>, 'timbre': <function timbre_metric>})*

The built-in metrics, by name.

### denote.compare.Metric

an aligned pair -> a distance (lower is closer).

* **Type:**
  A metric

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`AlignedPair`](#denote.compare.AlignedPair)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### denote.compare.chroma_metric(pair)

The alignment’s chroma cost: mean cosine distance along the path.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### denote.compare.compare_audio(audio, reference, \*, metrics=('chroma', 'timbre'), registry=mappingproxy({'chroma': <function chroma_metric>, 'timbre': <function timbre_metric>}), sr=None, reference_sr=None, subsequence=False, transpose=0, frame_rate=10.0, audio_sr=22050)

How close `audio` is to `reference`, metric by metric, on the same musical moments.

* **Parameters:**
  * **audio** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – The version to judge: a path, or an array with `sr`.
  * **reference** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – The reference: a path, or an array with `reference_sr`.
  * **metrics** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`AlignedPair`](#denote.compare.AlignedPair)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Metric names looked up in `registry`, or a `{name: metric}`
    mapping used as is (to add one without registering it).
  * **registry** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`AlignedPair`](#denote.compare.AlignedPair)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Where names are looked up (default [`METRICS`](#denote.compare.METRICS)).
  * **subsequence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – As for
    [`align_audio()`](denote.align.html.md#denote.align.align_audio), which is run once and shared.
  * **transpose** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – As for
    [`align_audio()`](denote.align.html.md#denote.align.align_audio), which is run once and shared.
  * **frame_rate** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – As for
    [`align_audio()`](denote.align.html.md#denote.align.align_audio), which is run once and shared.
  * **audio_sr** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – As for
    [`align_audio()`](denote.align.html.md#denote.align.align_audio), which is run once and shared.
* **Return type:**
  [`AudioComparison`](#denote.compare.AudioComparison)
* **Returns:**
  An [`AudioComparison`](#denote.compare.AudioComparison).

Example:

```default
c = denote.compare_audio("render.wav", "original.wav")
c["chroma"], c["timbre"]
```

### denote.compare.timbre_metric(pair, , n_mfcc=20)

Mean RMS difference of pooled-z-scored MFCCs `c1..` along the path; see the module.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
