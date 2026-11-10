"""Every backend declares its weights licence and commercial use; backend_table reads them."""

import sys

import pytest

import denote
from denote.base import COMMERCIAL_USE_VALUES
from denote.registry import clear_registry, get_config, list_backends, register_backend


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
        ("pesto", "LGPL-3.0", "yes"),  # LGPL binds redistributing weights, not outputs
        ("basic_pitch", "Apache-2.0", "yes"),
        ("torchcrepe", "MIT", "yes"),
        ("torchaudio_hdemucs", "unstated", "unclear"),  # research use per the author
        ("demucs_mlx", "unstated", "unclear"),  # converted from the same checkpoints
        ("roformer", "MIT", "yes"),  # the default checkpoint's card
    ],
)
def test_known_licences(name, weights_license, commercial_use):
    config = get_config(name)
    assert config["weights_license"] == weights_license
    assert config["commercial_use"] == commercial_use


def test_backend_table_lists_every_backend_with_its_licence_fields():
    rows = denote.backend_table()
    assert [r["name"] for r in rows] == list_backends()
    assert all(tuple(r) == denote.BACKEND_TABLE_FIELDS for r in rows)
    assert {"beat_this", "swift_f0", "pesto"} <= {r["name"] for r in rows}
    assert {"demucs_mlx", "roformer"} <= {r["name"] for r in rows}


def test_backend_table_filters_by_task_and_selects_fields():
    rows = denote.backend_table("pitch", fields=("name", "commercial_use"))
    assert {r["name"] for r in rows} >= {"swift_f0", "pesto", "librosa_pyin"}
    assert all(set(r) == {"name", "commercial_use"} for r in rows)
    assert all("pitch" in get_config(r["name"])["tasks"] for r in rows)


def _is_adapter(module_name):
    return module_name.startswith("denote.backends.") and module_name.endswith(
        ".adapter"
    )


def test_backend_table_loads_no_adapter(monkeypatch):
    # Earlier tests may have imported adapters: forget them, so an import here shows.
    for module_name in [m for m in sys.modules if _is_adapter(m)]:
        monkeypatch.delitem(sys.modules, module_name)
    denote.backend_table()
    assert not [m for m in sys.modules if _is_adapter(m)]


def test_weights_license_is_a_short_token():
    for row in denote.backend_table(fields=("name", "weights_license")):
        assert len(row["weights_license"]) <= 12, row


def test_register_backend_validates_commercial_use():
    with pytest.raises(ValueError, match="commercial_use"):
        register_backend("bad", {"name": "bad", "tasks": [], "commercial_use": "maybe"})
    config = {"name": "plain", "tasks": ["pitch"]}
    register_backend("plain", config)
    assert get_config("plain")["commercial_use"] == "unclear"
    assert "commercial_use" not in config  # the caller's dict is untouched
    register_backend("hosted", {**config, "commercial_use": "provider terms"})
    assert get_config("hosted")["commercial_use"] == "provider terms"
