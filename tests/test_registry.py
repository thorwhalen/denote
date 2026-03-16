"""Tests for denote.registry — backend discovery and registration."""

import pytest

from denote.registry import (
    clear_registry,
    register_backend,
    list_backends,
    get_default_backend,
    get_config,
    _discover_backends,
)


@pytest.fixture(autouse=True)
def fresh_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


def test_discover_finds_builtin_backends():
    _discover_backends()
    backends = list_backends()
    # At minimum, librosa backends should always be found
    assert 'librosa_pyin' in backends
    assert 'librosa_beats' in backends


def test_list_backends_by_task():
    _discover_backends()
    pitch_backends = list_backends('pitch')
    assert 'librosa_pyin' in pitch_backends
    beat_backends = list_backends('beats')
    assert 'librosa_beats' in beat_backends


def test_register_custom_backend():
    config = {
        'name': 'my_backend',
        'tasks': ['transcribe'],
        'default_for': [],
    }

    class FakeAdapter:
        def transcribe(self, audio, **kw):
            return 'fake'

    register_backend('my_backend', config, adapter=FakeAdapter())
    assert 'my_backend' in list_backends()
    assert 'my_backend' in list_backends('transcribe')


def test_get_default_backend_pitch():
    _discover_backends()
    default = get_default_backend('pitch')
    # torchcrepe is default if registered, otherwise librosa_pyin
    assert default in ('torchcrepe', 'librosa_pyin')


def test_get_default_backend_beats():
    _discover_backends()
    default = get_default_backend('beats')
    assert default == 'librosa_beats'


def test_get_config():
    _discover_backends()
    config = get_config('librosa_pyin')
    assert config['name'] == 'librosa_pyin'
    assert 'pitch' in config['tasks']


def test_unknown_backend_raises():
    _discover_backends()
    with pytest.raises(KeyError, match='Unknown backend'):
        get_config('nonexistent_backend')


def test_no_backends_for_task_raises():
    # Empty registry, no backends at all
    with pytest.raises(ValueError, match='No backends registered'):
        get_default_backend('separate')
