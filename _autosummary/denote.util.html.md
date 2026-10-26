# denote.util

Audio loading (any layout to `(channels, frames)`), and chunked overlap-add for models
that take bounded lengths (`chunked_apply`).

### Functions

| [`channels_first`](#denote.util.channels_first)(y)                            | A multichannel array as `(channels, frames)`, whichever way it came.                                                                        |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| [`chunk_bounds`](#denote.util.chunk_bounds)(n, \*, segment, overlap)        | `(start, end)` of the chunks that cover `n` frames, `segment` long and overlapping by at least `overlap`.                                   |
| [`chunked_apply`](#denote.util.chunked_apply)(fn, x, \*, segment, overlap)   | Apply `fn` to overlapping chunks of `x` along its last axis and overlap-add the results, so `fn` only ever sees `segment` frames at a time. |
| [`ensure_file_path`](#denote.util.ensure_file_path)(audio, \*[, sr])            | Ensure audio is available as a file path.                                                                                                   |
| [`load_audio`](#denote.util.load_audio)(audio, \*[, sr, mono, target_sr]) | Load audio from a file path or validate an existing array.                                                                                  |

### denote.util.channels_first(y)

A multichannel array as `(channels, frames)`, whichever way it came.

librosa and torchaudio give `(channels, frames)`, soundfile
`(frames, channels)`. Leading size-1 axes (a batch of one) are dropped. An
axis of size 2 or less beside a longer one is the channels axis, so a
one-frame stereo clip stays stereo; otherwise the longer axis is time.

* **Return type:**
  `ndarray`

### denote.util.chunk_bounds(n, , segment, overlap)

`(start, end)` of the chunks that cover `n` frames, `segment` long and
overlapping by at least `overlap`.

Every chunk is exactly `segment` long (one chunk of `n` when `n <= segment`):
the last one ends at `n`, so it may overlap its neighbour by more than
`overlap`.

```pycon
>>> chunk_bounds(10, segment=4, overlap=1)
[(0, 4), (3, 7), (6, 10)]
>>> chunk_bounds(9, segment=4, overlap=0)
[(0, 4), (4, 8), (5, 9)]
>>> chunk_bounds(3, segment=4, overlap=1)
[(0, 3)]
```

* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### denote.util.chunked_apply(fn, x, , segment, overlap)

Apply `fn` to overlapping chunks of `x` along its last axis and overlap-add
the results, so `fn` only ever sees `segment` frames at a time.

* **Parameters:**
  * **fn** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`ndarray`], `ndarray`]) – Takes a `(..., m)` chunk of `x` and returns a `(..., m)` array (any
    leading shape, the same for every chunk; `m` frames, as given).
  * **x** (`ndarray`) – Array, typically `(channels, frames)`.
  * **segment** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Chunk length in frames. Every chunk has this length unless `x` is
    shorter (see `chunk_bounds`).
  * **overlap** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum overlap between neighbouring chunks, in frames (0 allowed).
* **Returns:**
  where chunks overlap, a weighted average of their outputs,
  crossfaded linearly over the actual overlap. The output is divided by the
  summed weights, so it is an average even where ramps collide (an overlap
  above half the segment); the ramps alone sum to one wherever exactly two
  chunks overlap and do not collide.
* **Return type:**
  `ndarray`

Only the output, a `(frames,)` weight vector and one chunk’s temporaries are
held in memory; the division is done in place.

```pycon
>>> x = np.arange(10.0)[None]
>>> np.allclose(chunked_apply(lambda c: 2 * c, x, segment=4, overlap=1), 2 * x)
True
```

### denote.util.ensure_file_path(audio, , sr=None)

Ensure audio is available as a file path.

If audio is already a path, return it as a string.
If it’s an array, write to a temporary WAV file and return the path.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.util.load_audio(audio, , sr=None, mono=True, target_sr=None)

Load audio from a file path or validate an existing array.

* **Parameters:**
  * **audio** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – File path (str/Path) or numpy array of audio samples.
  * **sr** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Sample rate. Required when audio is an array. Ignored for file paths
    (detected automatically).
  * **mono** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, convert to mono.
  * **target_sr** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – If set, resample to this rate.
* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]
* **Returns:**
  Tuple of (audio_array, sample_rate).
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If audio is an array and sr is not provided.
  * [**FileNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#FileNotFoundError) – If audio is a path that doesn’t exist.
