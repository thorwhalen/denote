"""Every backend declares its weights licence and commercial use; backend_table reads them."""

import sys

import pytest

import denote
from denote.base import COMMERCIAL_USE_VALUES
from denote.registry import clear_registry, get_config, list_backends


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.mark.parametrize("name", list_backends())
def test_every_backend_declares_weights_license_and_commercial_use(name):
    config = get_config(name)
    assert isinstance(config.get("weights_license"), str) and config["weights_license"]
    assert config.get("commercial_use") in COMMERCIAL_USE_VALUES


@pytest.mark.parametrize(
    "name, weights_license, commercial_use",
    [
        ("beat_this", "MIT", "yes"),
        ("swift_f0", "MIT", "yes"),
        ("pesto", "LGPL-3.0", "conditional"),
        ("basic_pitch", "Apache-2.0", "yes"),
        ("torchcrepe", "MIT", "yes"),
        ("torchaudio_hdemucs", None, "unclear"),  # the htdemucs-family weights' terms
    ],
)
def test_known_licences(name, weights_license, commercial_use):
    config = get_config(name)
    if weights_license is not None:
        assert config["weights_license"] == weights_license
    assert config["commercial_use"] == commercial_use


def test_backend_table_lists_every_backend_with_its_licence_fields():
    rows = denote.backend_table()
    assert [r["name"] for r in rows] == list_backends()
    assert all(tuple(r) == denote.BACKEND_TABLE_FIELDS for r in rows)
    assert {"beat_this", "swift_f0", "pesto"} <= {r["name"] for r in rows}


def test_backend_table_filters_by_task_and_selects_fields():
    rows = denote.backend_table("pitch", fields=("name", "commercial_use"))
    assert {r["name"] for r in rows} >= {"swift_f0", "pesto", "librosa_pyin"}
    assert all(set(r) == {"name", "commercial_use"} for r in rows)
    assert all("pitch" in get_config(r["name"])["tasks"] for r in rows)


def test_backend_table_loads_no_adapter():
    before = set(sys.modules)
    denote.backend_table()
    loaded = {m for m in set(sys.modules) - before if m.endswith(".adapter")}
    assert not loaded
