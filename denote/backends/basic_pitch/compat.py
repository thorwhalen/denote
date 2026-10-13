"""Make basic-pitch importable and runnable against today's dependencies.

basic-pitch pins old dependencies, and on Python >= 3.11 off macOS every release
also requires a TensorFlow with no cp312 wheel. So the version you can actually
get is often 0.3.0, run against current scipy, setuptools and stray TensorFlow
leftovers. Three things then break (issue #2):

1. ``import tensorflow`` succeeds on an empty ``tensorflow/`` namespace folder (left
   by an uninstall), so basic-pitch sets ``TF_PRESENT`` and later fails on
   ``import tensorflow.lite``.
2. resampy < 0.4.3, which basic-pitch pins, imports ``pkg_resources``, removed
   from setuptools >= 81.
3. basic-pitch < 0.4 calls ``scipy.signal.gaussian``, removed in scipy >= 1.13
   (now ``scipy.signal.windows.gaussian``).

Each fix below is applied only when its failure condition holds, and each is scoped
as narrowly as the failure allows. Nothing is patched globally that a working
install would see: a real TensorFlow is left alone, ``pkg_resources`` is stubbed
only while resampy is imported, and the scipy alias is visible to basic-pitch's
``note_creation`` module only.

Use :func:`import_basic_pitch` rather than importing ``basic_pitch`` directly.
"""

import importlib
import importlib.resources
import importlib.util
import sys
import types
from pathlib import Path

#: Model runtimes basic-pitch can use, by import name, and the pip name to suggest.
MODEL_RUNTIMES = {
    "onnxruntime": "onnxruntime",
    "tflite_runtime": "tflite-runtime",
    "tensorflow": "tensorflow",
    "coremltools": "coremltools",
}

#: The ONNX serialization of the ICASSP 2022 model, relative to the basic_pitch package.
ONNX_MODEL_RELPATH = Path("saved_models", "icassp_2022", "nmp.onnx")


def is_namespace_leftover(name: str) -> bool:
    """True if ``name`` is importable only as an empty namespace package.

    That is what an uninstall leaves behind when stray files kept the folder: the
    import succeeds, but nothing in it works.

    >>> is_namespace_leftover('json')
    False
    >>> is_namespace_leftover('surely_not_an_installed_module_xyz')
    False
    """
    if sys.modules.get(name) is not None:
        return False
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin in (None, "namespace")


def _is_namespace_module(module: types.ModuleType) -> bool:
    spec = getattr(module, "__spec__", None)
    return getattr(module, "__file__", None) is None and (
        spec is None or spec.origin in (None, "namespace")
    )


def hide_broken_tensorflow() -> bool:
    """Make ``import tensorflow`` fail if TensorFlow is only an empty folder.

    basic-pitch decides at import time whether TensorFlow is present from a bare
    ``import tensorflow``. If the leftover was already imported (say, by an earlier
    ``import basic_pitch``), it is evicted and basic-pitch's ``TF_PRESENT`` reset,
    which still takes effect as long as ``basic_pitch.inference`` is not yet
    imported. Returns True if it hid a leftover.
    """
    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        if tf is None or not _is_namespace_module(tf):
            return False
    elif not is_namespace_leftover("tensorflow"):
        return False
    sys.modules["tensorflow"] = None  # makes any `import tensorflow...` raise
    basic_pitch = sys.modules.get("basic_pitch")
    if getattr(basic_pitch, "TF_PRESENT", False):
        basic_pitch.TF_PRESENT = False
    return True


def _resource_filename(package: str, resource: str) -> str:
    return str(importlib.resources.files(package) / resource)


def preload_resampy() -> bool:
    """Import resampy even when ``pkg_resources`` is gone (setuptools >= 81).

    resampy < 0.4.3 uses only ``pkg_resources.resource_filename``. A stub providing
    it is put in ``sys.modules`` for the duration of resampy's import and removed
    after, so no other library ever sees a partial ``pkg_resources``. Returns True
    if the stub was needed.
    """
    if "resampy" in sys.modules or importlib.util.find_spec("resampy") is None:
        return False
    if importlib.util.find_spec("pkg_resources") is not None:
        return False
    stub = types.ModuleType("pkg_resources")
    stub.resource_filename = _resource_filename
    sys.modules["pkg_resources"] = stub
    try:
        importlib.import_module("resampy")
    finally:
        if sys.modules.get("pkg_resources") is stub:
            del sys.modules["pkg_resources"]
    return True


def _module_proxy(module: types.ModuleType, **overrides) -> types.ModuleType:
    """A module that delegates to ``module`` except for ``overrides``."""
    proxy = types.ModuleType(module.__name__)
    proxy.__dict__.update(overrides)
    proxy.__getattr__ = lambda name: getattr(module, name)  # PEP 562
    return proxy


def restore_scipy_gaussian(note_creation: types.ModuleType) -> bool:
    """Give ``note_creation`` a ``scipy.signal.gaussian`` if scipy no longer has one.

    Only that module's ``scipy`` name is replaced, by a proxy of the real scipy, so
    scipy itself is not modified. Returns True if the alias was needed.
    """
    scipy = note_creation.scipy
    if hasattr(scipy.signal, "gaussian"):
        return False
    signal = _module_proxy(scipy.signal, gaussian=scipy.signal.windows.gaussian)
    note_creation.scipy = _module_proxy(scipy, signal=signal)
    return True


def _check_model_runtime():
    present = [
        name
        for name in MODEL_RUNTIMES
        if sys.modules.get(name) is not None
        or (
            name not in sys.modules
            and importlib.util.find_spec(name) is not None
            and not is_namespace_leftover(name)
        )
    ]
    if not present:
        raise ImportError(
            "basic-pitch needs a model runtime and none is installed. The portable "
            "one is ONNX: pip install onnxruntime  (or: pip install 'denote[basic_pitch]')"
        )


def model_path(basic_pitch: types.ModuleType) -> Path:
    """The model file to load: ONNX when onnxruntime is usable, else basic-pitch's default.

    ONNX loads the same everywhere, unlike the TensorFlow SavedModel, whose loader
    breaks across TensorFlow versions.
    """
    onnx_path = Path(basic_pitch.__file__).parent / ONNX_MODEL_RELPATH
    if getattr(basic_pitch, "ONNX_PRESENT", False) and onnx_path.exists():
        return onnx_path
    return Path(basic_pitch.ICASSP_2022_MODEL_PATH)


def import_basic_pitch():
    """Import basic-pitch with the fixes above; return ``(predict, model_path)``."""
    hide_broken_tensorflow()
    _check_model_runtime()
    preload_resampy()
    import basic_pitch
    import basic_pitch.note_creation
    from basic_pitch.inference import predict

    restore_scipy_gaussian(basic_pitch.note_creation)
    return predict, model_path(basic_pitch)
