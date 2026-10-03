# denote.util

Audio loading and utility functions for denote.

### Functions

| [`ensure_file_path`](#denote.util.ensure_file_path)(audio, \*[, sr])            | Ensure audio is available as a file path.                  |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`load_audio`](#denote.util.load_audio)(audio, \*[, sr, mono, target_sr]) | Load audio from a file path or validate an existing array. |

### denote.util.ensure_file_path(audio, , sr=None)

Ensure audio is available as a file path.

If audio is already a path, return it as a string.
If it’s an array, write to a temporary WAV file and return the path.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.util.load_audio(audio, , sr=None, mono=True, target_sr=None)

Load audio from a file path or validate an existing array.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), `ndarray`]) – File path (str/Path) or numpy array of audio samples.
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
