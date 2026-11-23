# denote.backends.chroma_hmm.adapter

Adapter for the model-free chord recognizer in `denote.chords`.

### Classes

| [`Adapter`](#denote.backends.chroma_hmm.adapter.Adapter)(config)   | Chords as `[{start, end, label}]` in Harte syntax, free, vocabulary-restricted or aligned.   |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|

### *class* denote.backends.chroma_hmm.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Chords as `[{start, end, label}]` in Harte syntax, free, vocabulary-restricted or aligned.

#### get_chords(audio, , sr=None, \*\*kwargs)

Recognize the chords of `audio`.

* **Parameters:**
  * **audio** – File path or numpy array.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – `vocabulary` (restrict the chords to this list), `sequence` (an ordered
    chord list the audio is known to follow, repeats allowed: returns when each
    happens, one segment per entry, `raw['positions']` giving the entry index;
    excludes `vocabulary` and `causal`), `qualities` (default vocabulary’s
    qualities: `("maj", "min")`; add `"7"`, `"maj7"`, `"min7"`…),
    `causal`, `dwell_s`, `kappa`, `harmonics`, `gate_db`, `no_chord_cos`, `window_s`,
    `hop_s`, `audio_sr`, `frontend` (`"peaks"` default, `"basic_pitch"`, or a callable;
    see `denote.chroma_frontends`). With `sequence`: `sections` (one key per entry; the order
    becomes tolerant: jumps between section starts at `jump_p`, skipped entries at
    `skip_p`).
* **Returns:**
  `intervals` (N, 2) seconds, Harte `labels` (`N` = no chord),
  `segments()` for `[{start, end, label}]`; `raw` holds the per-frame path,
  states and chroma.
* **Return type:**
  [*ChordResult*](denote.base.html.md#denote.base.ChordResult)
