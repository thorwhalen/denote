"""Tests for denote.services — ServiceCollection and ServiceHandle."""

import pytest

from denote.registry import clear_registry, register_backend, _discover_backends
from denote.services import ServiceCollection


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_registry()
    yield
    clear_registry()


class FakeAdapter:
    def __init__(self, config):
        self.config = config

    def transcribe(self, audio, **kw):
        return f'transcribed:{audio}'

    def get_pitch(self, audio, **kw):
        return f'pitch:{audio}'


def _register_fake():
    config = {
        'name': 'fake',
        'display_name': 'Fake Backend',
        'tasks': ['transcribe', 'pitch'],
        'default_for': ['transcribe'],
        'license': 'MIT',
        'pip_install': 'fake-backend',
    }
    register_backend('fake', config, adapter=FakeAdapter(config))


def test_service_collection_contains():
    _register_fake()
    services = ServiceCollection()
    assert 'fake' in services


def test_service_collection_getitem():
    _register_fake()
    services = ServiceCollection()
    handle = services['fake']
    assert handle.name == 'fake'


def test_service_collection_getattr():
    _register_fake()
    services = ServiceCollection()
    handle = services.fake
    assert handle.name == 'fake'


def test_service_collection_iter():
    _register_fake()
    services = ServiceCollection()
    assert 'fake' in list(services)


def test_service_collection_len():
    _register_fake()
    services = ServiceCollection()
    assert len(services) >= 1


def test_service_handle_proxies_methods():
    _register_fake()
    services = ServiceCollection()
    result = services.fake.transcribe('test.wav')
    assert result == 'transcribed:test.wav'


def test_service_handle_info():
    _register_fake()
    services = ServiceCollection()
    info = services.fake.info
    assert info['name'] == 'fake'
    assert 'transcribe' in info['tasks']


def test_unknown_backend_raises():
    services = ServiceCollection()
    with pytest.raises(KeyError):
        services['nonexistent']


def test_unknown_attr_raises():
    services = ServiceCollection()
    with pytest.raises(AttributeError):
        services.nonexistent
