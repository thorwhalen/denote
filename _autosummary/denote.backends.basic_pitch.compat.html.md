# denote.backends.basic_pitch.compat

Make basic-pitch importable and runnable against today’s dependencies.

basic-pitch pins old dependencies. On Python 3.12, pip resolves only 0.3.0 on
macOS, and no release at all on Linux (each pins a TensorFlow with no cp312
wheel), so it gets installed with `--no-deps`. Either way it runs against current
scipy, setuptools and stray TensorFlow leftovers. Three things then break (issue #2):

1. `import tensorflow` succeeds on an empty `tensorflow/` namespace folder (left
   by an uninstall), so basic-pitch sets `TF_PRESENT` and later fails on
   `import tensorflow.lite`.
2. resampy < 0.4.3, which basic-pitch pins, imports `pkg_resources`, removed
   from setuptools >= 81.
3. basic-pitch < 0.4 calls `scipy.signal.gaussian`, removed in scipy >= 1.13
   (now `scipy.signal.windows.gaussian`).

Each fix below is applied only when its failure condition holds, and each is scoped
as narrowly as the failure allows. Nothing is patched globally that a working
install would see: a real TensorFlow is left alone, `pkg_resources` is stubbed
only while resampy is imported, and the scipy alias is visible to basic-pitch’s
`note_creation` module only.

Use [`import_basic_pitch()`](#denote.backends.basic_pitch.compat.import_basic_pitch) rather than importing `basic_pitch` directly.

### Module Attributes

| [`MODEL_RUNTIMES`](#denote.backends.basic_pitch.compat.MODEL_RUNTIMES)     | Model runtimes basic-pitch can use, by import name, and the pip name to suggest.      |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`ONNX_MODEL_RELPATH`](#denote.backends.basic_pitch.compat.ONNX_MODEL_RELPATH) | The ONNX serialization of the ICASSP 2022 model, relative to the basic_pitch package. |

### Functions

| [`hide_broken_tensorflow`](#denote.backends.basic_pitch.compat.hide_broken_tensorflow)()              | Make `import tensorflow` fail if TensorFlow is only an empty folder.                 |
|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`import_basic_pitch`](#denote.backends.basic_pitch.compat.import_basic_pitch)()                  | Import basic-pitch with the fixes above; return `(predict, model_path)`.             |
| [`is_namespace_leftover`](#denote.backends.basic_pitch.compat.is_namespace_leftover)(name)           | True if `name` is importable only as an empty namespace package.                     |
| [`model_path`](#denote.backends.basic_pitch.compat.model_path)(basic_pitch)               | The model file to load: ONNX when onnxruntime is usable, else basic-pitch's default. |
| [`preload_resampy`](#denote.backends.basic_pitch.compat.preload_resampy)()                     | Import resampy even when `pkg_resources` is gone (setuptools >= 81).                 |
| [`restore_scipy_gaussian`](#denote.backends.basic_pitch.compat.restore_scipy_gaussian)(note_creation) | Give `note_creation` a `scipy.signal.gaussian` if scipy no longer has one.           |

### denote.backends.basic_pitch.compat.MODEL_RUNTIMES *= {'coremltools': 'coremltools', 'onnxruntime': 'onnxruntime', 'tensorflow': 'tensorflow', 'tflite_runtime': 'tflite-runtime'}*

Model runtimes basic-pitch can use, by import name, and the pip name to suggest.

### denote.backends.basic_pitch.compat.ONNX_MODEL_RELPATH *= PosixPath('saved_models/icassp_2022/nmp.onnx')*

The ONNX serialization of the ICASSP 2022 model, relative to the basic_pitch package.

### denote.backends.basic_pitch.compat.hide_broken_tensorflow()

Make `import tensorflow` fail if TensorFlow is only an empty folder.

basic-pitch decides at import time whether TensorFlow is present from a bare
`import tensorflow`. If the leftover was already imported (say, by an earlier
`import basic_pitch`), it is evicted and basic-pitch’s `TF_PRESENT` reset,
which still takes effect as long as `basic_pitch.inference` is not yet
imported. Returns True if it hid a leftover.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### denote.backends.basic_pitch.compat.import_basic_pitch()

Import basic-pitch with the fixes above; return `(predict, model_path)`.

### denote.backends.basic_pitch.compat.is_namespace_leftover(name)

True if `name` is importable only as an empty namespace package.

That is what an uninstall leaves behind when stray files kept the folder: the
import succeeds, but nothing in it works.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> is_namespace_leftover('json')
False
>>> is_namespace_leftover('surely_not_an_installed_module_xyz')
False
```

### denote.backends.basic_pitch.compat.model_path(basic_pitch)

The model file to load: ONNX when onnxruntime is usable, else basic-pitch’s default.

ONNX loads the same everywhere, unlike the TensorFlow SavedModel, whose loader
breaks across TensorFlow versions.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### denote.backends.basic_pitch.compat.preload_resampy()

Import resampy even when `pkg_resources` is gone (setuptools >= 81).

resampy < 0.4.3 uses only `pkg_resources.resource_filename`. A stub providing
it is put in `sys.modules` for the duration of resampy’s import and removed
after, so no other library ever sees a partial `pkg_resources`. Returns True
if the stub was needed.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### denote.backends.basic_pitch.compat.restore_scipy_gaussian(note_creation)

Give `note_creation` a `scipy.signal.gaussian` if scipy no longer has one.

Only that module’s `scipy` name is replaced, by a proxy of the real scipy, so
scipy itself is not modified. Returns True if the alias was needed.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
