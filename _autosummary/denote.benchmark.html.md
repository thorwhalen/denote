# denote.benchmark

Score chord recognition against a labelled set with `mir_eval`, the same way for any backend.

A labelled set is an iterable of `(audio, reference)` pairs (or `ChordItem`); a reference is
`(intervals, labels)` or `[{start, end, label}]`, labels in Harte syntax. Scores are
`mir_eval.chord` weighted chord symbol recall (`root`, `majmin`, `sevenths`, …) plus
segmentation quality, per item and pooled (duration-weighted over the whole set).

```pycon
>>> from denote.benchmark import ChordItem, evaluate_chords
>>> report = evaluate_chords(items)
>>> report["pooled"]["majmin"]
0.71
```

`guitarset_items(root)` yields GuitarSet’s microphone comping takes (needs `jams`);
`load_jams_chords` reads any JAMS chord annotation. `mir_eval` is imported lazily and comes
with the `eval` extra.

### Module Attributes

| [`DFLT_METRICS`](#denote.benchmark.DFLT_METRICS)   | mir_eval.chord comparison functions scored by default, loosest to strictest   |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------|

### Functions

| [`evaluate_chords`](#denote.benchmark.evaluate_chords)(items, \*[, backend, ...])       | Run a chord recogniser over a labelled set and score it.                                    |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`guitarset_items`](#denote.benchmark.guitarset_items)(root, \*[, players, annotation]) | GuitarSet microphone comping takes as <br/><br/>```<br/>``<br/>```<br/><br/>ChordItem\`\`s. |
| [`load_jams_chords`](#denote.benchmark.load_jams_chords)(path, \*[, index, namespace])   | `(intervals, labels)` of the `index`-th chord annotation of a JAMS file (needs `jams`).     |
| [`score_chords`](#denote.benchmark.score_chords)(reference, estimate, \*[, ...])     | Score one estimate against one reference.                                                   |

### Classes

| [`ChordItem`](#denote.benchmark.ChordItem)(audio, reference[, name, sr, kwargs])   | One labelled recording: audio (path or array + `sr`) and its reference chords.   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|

### *class* denote.benchmark.ChordItem(audio, reference, name='', sr=None, kwargs=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One labelled recording: audio (path or array + `sr`) and its reference chords.

### denote.benchmark.DFLT_METRICS *= ('root', 'majmin', 'sevenths')*

mir_eval.chord comparison functions scored by default, loosest to strictest

### denote.benchmark.evaluate_chords(items, , backend=None, recognizer=None, metrics=('root', 'majmin', 'sevenths'), \*\*backend_kwargs)

Run a chord recogniser over a labelled set and score it.

* **Parameters:**
  * **items** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – `(audio, reference)` pairs or `ChordItem`.
  * **backend** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A chords backend name (default: the registry’s default for chords).
  * **recognizer** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Instead of a backend, any `f(audio, **kwargs) -> ChordResult`.
  * **metrics** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – `mir_eval.chord` comparison names.
  * **\*\*backend_kwargs** – passed to every call (e.g. `vocabulary=`, `causal=True`).
* **Returns:**
  {name: {metric: ratio, …, “segmentation”: …}}, “pooled”: {metric: ratio},
  “n_items”: n, “backend”: name}\`\`. Pooled ratios weight each item by its duration.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### denote.benchmark.guitarset_items(root, , players=None, annotation='performed')

GuitarSet microphone comping takes as 

```
``
```

ChordItem\`\`s.

`root` holds `audio_mono-mic/*_comp_mic.wav` and `ann/*.jams` (the dataset’s layout,
CC BY 4.0). `players` restricts to those player numbers 0-5; `annotation` is `"performed"`
(what was played) or `"instructed"` (the lead sheet).

### denote.benchmark.load_jams_chords(path, , index=0, namespace='chord')

`(intervals, labels)` of the `index`-th chord annotation of a JAMS file (needs `jams`).

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]

### denote.benchmark.score_chords(reference, estimate, , duration=None, metrics=('root', 'majmin', 'sevenths'))

Score one estimate against one reference. Both are `(intervals, labels)` or segment lists.

Returns `{metric: (credit_seconds, total_seconds), ..., "segmentation": {...}}`: seconds, not
ratios, so a set can be pooled by summing. Time outside either annotation counts as `N`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
