"""Tests for denote.keys: profiles, key names, and correlation scoring (no audio)."""

import numpy as np
import pytest

from denote.keys import (
    KEY_PROFILES,
    KeyProfile,
    key_from_chroma,
    key_name,
    key_scores,
    parse_key,
    relative_key,
)


@pytest.mark.parametrize("name", sorted(KEY_PROFILES))
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_each_rotated_profile_recovers_its_key(name, mode):
    weights = getattr(KEY_PROFILES[name], mode)
    for tonic in range(12):
        result = key_from_chroma(np.roll(weights, tonic), profile=name)
        assert result.key == key_name(tonic, mode)
        assert result.confidence == pytest.approx(1.0)
        assert result.margin > 0
        assert result.profile == name


def test_scores_are_ranked_and_complete():
    result = key_from_chroma(np.roll(KEY_PROFILES["krumhansl"].major, 7))
    values = list(result.scores.values())
    assert len(values) == 24
    assert values == sorted(values, reverse=True)
    assert result.ranked[0] == (result.key, result.confidence)
    assert result.runner_up == result.ranked[1][0]
    assert result.margin == pytest.approx(values[0] - values[1])


@pytest.mark.parametrize("name", sorted(KEY_PROFILES))
def test_relative_keys_tie_on_bare_pitch_class_sets(name):
    """With no tonic weighting, a key's relative is the runner-up."""
    white_keys = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    result = key_from_chroma(white_keys, profile=name)
    assert result.key == "C major"
    assert result.runner_up == result.relative == "A minor"


def test_chromagram_is_averaged_over_frames():
    vector = np.asarray(KEY_PROFILES["krumhansl"].minor)
    chromagram = np.stack([vector * 0.5, vector * 1.5], axis=1)  # (12, 2)
    assert key_from_chroma(chromagram).key == "C minor"


def test_custom_profile_pair():
    major = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # diatonic membership
    minor = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]  # harmonic minor
    result = key_from_chroma(np.roll(minor, 4), profile=(major, minor))
    assert result.key == "E minor"
    assert result.profile == "custom"
    assert (
        key_from_chroma(major, profile=KeyProfile(tuple(major), tuple(minor))).key
        == "C major"
    )


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="flat"):
        key_scores(np.ones(12))
    with pytest.raises(ValueError, match="shape"):
        key_scores(np.ones(11))
    with pytest.raises(ValueError, match="unknown key profile"):
        key_scores(np.arange(12), profile="nope")
    with pytest.raises(ValueError, match="12 weights"):
        key_scores(np.arange(12), profile=([1, 2], [3, 4]))


def test_key_names_round_trip():
    for mode in ("major", "minor"):
        for pc in range(12):
            assert parse_key(key_name(pc, mode)) == (pc, mode)
    assert parse_key("D#m") == (3, "minor")
    assert parse_key("B♭ maj") == (10, "major")
    assert relative_key("Eb major") == "C minor"
    assert relative_key("C# minor") == "E major"
    for bad in ("H major", "C dorian", "", "C# x y"):
        with pytest.raises(ValueError):
            parse_key(bad)


def test_profiles_match_music21():
    """A second published source for the numbers (music21 copies Humdrum keycor)."""
    discrete = pytest.importorskip("music21.analysis.discrete")
    pairs = {
        "krumhansl": discrete.KrumhanslKessler(),
        "temperley": discrete.TemperleyKostkaPayne(),
    }
    for name, analysis in pairs.items():
        for mode in ("major", "minor"):
            assert getattr(KEY_PROFILES[name], mode) == tuple(analysis.getWeights(mode))


def test_spelling_matches_chordsym():
    """Tonics are spelled the way chordsym spells keys."""
    keys = pytest.importorskip("chordsym.keys")
    from chordsym.transpose import key_name as chordsym_key_name

    for pc in range(12):
        assert chordsym_key_name(keys.Key(pc)) == key_name(pc, "major").split()[0]
        assert (
            chordsym_key_name(keys.Key(pc, minor=True))
            == key_name(pc, "minor").split()[0] + "m"
        )
