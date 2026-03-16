"""Tests for denote.translation — parameter normalization."""

import warnings
import pytest

from denote.translation import make_kwargs_translator, validate_param


def test_basic_translation():
    param_map = {
        'min_frequency': {'native_name': 'fmin', 'default': 50.0},
        'max_frequency': {'native_name': 'fmax', 'default': 2000.0},
    }
    translate = make_kwargs_translator(param_map)
    result = translate(min_frequency=100.0)
    assert result == {'fmin': 100.0, 'fmax': 2000.0}


def test_coerce_function():
    param_map = {
        'min_note_length': {
            'native_name': 'minimum_note_length',
            'coerce': lambda x: x * 1000,  # seconds -> ms
        },
    }
    translate = make_kwargs_translator(param_map)
    result = translate(min_note_length=0.128)
    assert result['minimum_note_length'] == pytest.approx(128.0)


def test_unsupported_param_warns():
    param_map = {'min_frequency': {'native_name': 'fmin'}}
    translate = make_kwargs_translator(param_map, on_unsupported='warn')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = translate(min_frequency=50.0, unknown_param=42)
        assert len(w) == 1
        assert 'unknown_param' in str(w[0].message)
    assert 'unknown_param' not in result


def test_unsupported_param_raises():
    param_map = {'min_frequency': {'native_name': 'fmin'}}
    translate = make_kwargs_translator(param_map, on_unsupported='raise')
    with pytest.raises(ValueError, match='Unsupported parameter'):
        translate(unknown_param=42)


def test_none_config_marks_unsupported():
    param_map = {
        'device': None,
        'min_frequency': {'native_name': 'fmin'},
    }
    translate = make_kwargs_translator(param_map, on_unsupported='warn')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = translate(device='cuda', min_frequency=50.0)
        assert len(w) == 1
    assert 'device' not in result


def test_validate_param_min():
    with pytest.raises(ValueError, match='below minimum'):
        validate_param('threshold', -0.1, {'min': 0.0, 'max': 1.0})


def test_validate_param_max():
    with pytest.raises(ValueError, match='above maximum'):
        validate_param('threshold', 1.5, {'min': 0.0, 'max': 1.0})


def test_validate_param_choices():
    with pytest.raises(ValueError, match='not in'):
        validate_param('model', 'medium', {'choices': ['full', 'tiny']})
