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


def test_minor_profile_ranks_its_relative_second():
    scores = key_scores(KEY_PROFILES["krumhansl"].minor)
    assert list(scores)[:2] == ["C minor", "Eb major"]
    assert len(scores) == 24


def test_flat_within_tolerance_raises():
    with pytest.raises(ValueError, match="flat"):
        key_scores(np.ones(12) + 1e-12 * np.arange(12))
    with pytest.raises(ValueError, match="flat"):
        key_scores(np.zeros(12))


def test_weighted_average_over_frames():
    major = np.asarray(KEY_PROFILES["krumhansl"].major)
    chromagram = np.stack([major, np.roll(major, 6)], axis=1)
    assert key_from_chroma(chromagram, weights=[1.0, 0.0]).key == "C major"
    assert key_from_chroma(chromagram, weights=[0.0, 1.0]).key == "Gb major"
    with pytest.raises(ValueError, match="one value per frame"):
        key_from_chroma(chromagram, weights=[1.0])


def test_profile_label_for_objects():
    temperley = KEY_PROFILES["temperley"]
    assert key_from_chroma(temperley.major, profile=temperley).profile == "temperley"
    named = KeyProfile((1,) + (0,) * 11, (0,) * 11 + (1,), source="my corpus")
    assert key_from_chroma(np.arange(12), profile=named).profile == "my corpus"


def test_relative_is_empty_for_other_modes():
    from denote.base import KeyResult

    assert (
        KeyResult(key="D dorian", tonic="D", mode="dorian", confidence=0.5).relative
        == ""
    )


# Krumhansl (1990), Table 2.4: correlations between the probe-tone profiles of
# C major and other keys. These pin the published numbers without music21.
KK_INTERKEY = {
    ("major", 7): 0.591,
    ("minor", 9): 0.651,
    ("minor", 0): 0.511,
    ("minor", 4): 0.536,
}


@pytest.mark.parametrize("other, expected", list(KK_INTERKEY.items()))
def test_krumhansl_interkey_correlations(other, expected):
    mode, tonic = other
    kk = KEY_PROFILES["krumhansl"]
    c_major = np.asarray(kk.major)
    other_profile = np.roll(getattr(kk, mode), tonic)
    assert np.corrcoef(c_major, other_profile)[0, 1] == pytest.approx(
        expected, abs=0.002
    )


def test_profile_goldens():
    """Literal copies of the published tables (Krumhansl 1990; Temperley 2007)."""
    assert KEY_PROFILES["krumhansl"][:2] == (
        (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    )
    assert KEY_PROFILES["temperley"][:2] == (
        (
            0.748,
            0.060,
            0.488,
            0.082,
            0.670,
            0.460,
            0.096,
            0.715,
            0.104,
            0.366,
            0.057,
            0.400,
        ),
        (
            0.712,
            0.084,
            0.474,
            0.618,
            0.049,
            0.460,
            0.105,
            0.747,
            0.404,
            0.067,
            0.133,
            0.330,
        ),
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
    assert parse_key("Bb") == (10, "major")
    assert parse_key("c# minor") == (1, "minor")
    assert parse_key("am") == (9, "minor")
    assert parse_key("Amaj") == (9, "major")
    assert parse_key("Amin") == (9, "minor")
    assert parse_key("Bbmaj") == (10, "major")
    with pytest.raises(ValueError, match="ambiguous"):
        parse_key("a")  # lowercase can mean minor in some conventions
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
