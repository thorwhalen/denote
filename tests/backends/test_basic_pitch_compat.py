"""Unit tests for the basic-pitch compatibility fixes (issue #2).

None of these need basic-pitch installed: each fix is exercised against a fake of
the module it repairs.
"""

import importlib.util
import sys
import types

import numpy as np
import pytest

from denote.backends.basic_pitch import compat


@pytest.fixture
def forget_modules():
    """Drop the named modules from sys.modules after the test, restoring originals."""
    saved = {}

    def forget(*names):
        for name in names:
            saved.setdefault(name, sys.modules.pop(name, None))

    yield forget
    for name, module in saved.items():
        sys.modules.pop(name, None)
        if module is not None:
            sys.modules[name] = module


def test_namespace_leftover_is_detected(tmp_path, monkeypatch):
    (tmp_path / "leftover_pkg_xyz").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    assert compat.is_namespace_leftover("leftover_pkg_xyz")
    assert not compat.is_namespace_leftover("numpy")


def test_broken_tensorflow_is_hidden(tmp_path, monkeypatch, forget_modules):
    spec = importlib.util.find_spec("tensorflow")
    if spec is not None and spec.origin not in (None, "namespace"):
        pytest.skip("a real TensorFlow is installed here")
    (tmp_path / "tensorflow").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    forget_modules("tensorflow")

    assert compat.hide_broken_tensorflow()
    with pytest.raises(ImportError):
        import tensorflow.lite  # noqa: F401
    assert not compat.hide_broken_tensorflow()  # idempotent


def test_already_imported_leftover_is_evicted(tmp_path, monkeypatch, forget_modules):
    spec = importlib.util.find_spec("tensorflow")
    if spec is not None and spec.origin not in (None, "namespace"):
        pytest.skip("a real TensorFlow is installed here")
    (tmp_path / "tensorflow").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    forget_modules("tensorflow", "basic_pitch")
    import tensorflow  # noqa: F401  (the empty folder imports fine)

    basic_pitch = types.ModuleType("basic_pitch")
    basic_pitch.TF_PRESENT = True
    sys.modules["basic_pitch"] = basic_pitch

    assert compat.hide_broken_tensorflow()
    assert sys.modules["tensorflow"] is None
    assert basic_pitch.TF_PRESENT is False


def test_resampy_imports_without_pkg_resources(tmp_path, monkeypatch, forget_modules):
    fake = tmp_path / "resampy"
    fake.mkdir()
    (fake / "data.npy").write_bytes(b"")
    (fake / "__init__.py").write_text(
        "import pkg_resources\n"
        "DATA = pkg_resources.resource_filename(__name__, 'data.npy')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    forget_modules("resampy", "pkg_resources")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "pkg_resources" else real_find_spec(name, *a),
    )

    assert compat.preload_resampy()
    assert sys.modules["resampy"].DATA == str(fake / "data.npy")
    assert "pkg_resources" not in sys.modules  # the stub did not outlive the import
    assert not compat.preload_resampy()  # already imported


def _fake_scipy(*, with_gaussian):
    windows = types.SimpleNamespace(gaussian=lambda n, std: np.ones(n) * std)
    signal = types.ModuleType("scipy.signal")
    signal.windows = windows
    signal.argrelmax = lambda x, axis=0: ("argrelmax", axis)
    if with_gaussian:
        signal.gaussian = windows.gaussian
    scipy = types.ModuleType("scipy")
    scipy.signal = signal
    scipy.__version__ = "fake"
    return scipy


def test_scipy_gaussian_is_restored_for_note_creation_only():
    scipy = _fake_scipy(with_gaussian=False)
    note_creation = types.ModuleType("basic_pitch.note_creation")
    note_creation.scipy = scipy

    assert compat.restore_scipy_gaussian(note_creation)
    assert list(note_creation.scipy.signal.gaussian(3, std=5)) == [5, 5, 5]
    assert note_creation.scipy.signal.argrelmax([1], axis=0) == ("argrelmax", 0)
    assert note_creation.scipy.__version__ == "fake"
    assert not hasattr(scipy.signal, "gaussian")  # the real module is untouched


def test_scipy_with_gaussian_is_left_alone():
    scipy = _fake_scipy(with_gaussian=True)
    note_creation = types.ModuleType("basic_pitch.note_creation")
    note_creation.scipy = scipy
    assert not compat.restore_scipy_gaussian(note_creation)
    assert note_creation.scipy is scipy


def _fake_basic_pitch(tmp_path, *, onnx_present, onnx_file=True):
    model_dir = tmp_path / compat.ONNX_MODEL_RELPATH.parent
    model_dir.mkdir(parents=True)
    if onnx_file:
        (tmp_path / compat.ONNX_MODEL_RELPATH).write_bytes(b"")
    bp = types.ModuleType("basic_pitch")
    bp.__file__ = str(tmp_path / "__init__.py")
    bp.ONNX_PRESENT = onnx_present
    bp.ICASSP_2022_MODEL_PATH = model_dir / "nmp"
    return bp


def test_model_path_prefers_onnx_when_runtime_present(tmp_path):
    bp = _fake_basic_pitch(tmp_path, onnx_present=True)
    assert compat.model_path(bp) == tmp_path / compat.ONNX_MODEL_RELPATH


@pytest.mark.parametrize("onnx_present, onnx_file", [(False, True), (True, False)])
def test_model_path_falls_back_to_default(tmp_path, onnx_present, onnx_file):
    bp = _fake_basic_pitch(tmp_path, onnx_present=onnx_present, onnx_file=onnx_file)
    assert compat.model_path(bp) == bp.ICASSP_2022_MODEL_PATH


def test_missing_model_runtime_is_a_clear_error(monkeypatch, forget_modules):
    forget_modules(*compat.MODEL_RUNTIMES)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None
        if name in compat.MODEL_RUNTIMES
        else real_find_spec(name, *a),
    )
    with pytest.raises(ImportError, match="onnxruntime"):
        compat.import_basic_pitch()
