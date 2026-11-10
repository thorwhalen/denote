"""Tests for the librosa_key backend and the get_key facade, on synthetic progressions.

The fixture renders cadences (I-IV-V-I in major, i-iv-V-i with the harmonic-minor
dominant in minor) as sums of sine partials, so the expected key is known. The
raised leading tone in the minor dominant is what separates a minor key from its
relative major here: without it (natural minor), i-iv-v-i uses exactly the pitch
classes of the relative major, and the estimate would be legitimately ambiguous.
"""

import numpy as np
import pytest

SR = 22050

# chord tones as semitones above the tonic (root position, bass added below)
PROGRESSIONS = {
    "major": ((0, 4, 7), (5, 9, 12), (7, 11, 14), (0, 4, 7)),  # I IV V I
    "minor": ((0, 3, 7), (5, 8, 12), (7, 11, 14), (0, 3, 7)),  # i iv V i
}


def render_progression(
    tonic_pc, mode, *, sr=SR, chord_s=1.25, n_partials=4, base_midi=48
):
    """A cadence in the key (tonic_pc, mode): chords of decaying-partial tones + bass."""
    n = int(sr * chord_s)
    t = np.arange(n) / sr
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)  # 20 ms fades, no clicks

    def tone(midi):
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        return sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_partials + 1))

    root = base_midi + tonic_pc
    chords = []
    for chord in PROGRESSIONS[mode]:
        notes = [root + i for i in chord] + [root + chord[0] - 12]
        chords.append(fade * sum(tone(m) for m in notes))
    y = np.concatenate(chords)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


CASES = [
    (0, "major", "C major"),
    (7, "major", "G major"),
    (3, "major", "Eb major"),
    (9, "minor", "A minor"),
    (6, "minor", "F# minor"),
    (0, "minor", "C minor"),
]


@pytest.mark.parametrize("tonic_pc, mode, expected", CASES)
def test_get_key_on_cadences(tonic_pc, mode, expected):
    import denote
    from denote.base import KeyResult

    y = render_progression(tonic_pc, mode)
    result = denote.get_key(y, sr=SR)
    assert isinstance(result, KeyResult)
    assert result.key == expected, result.ranked[:3]
    assert result.backend == "librosa_key"
    assert result.profile == "krumhansl"
    assert result.margin > 0
    assert 0 < result.confidence <= 1
    assert len(result.scores) == 24
    assert result.raw["chroma"].shape == (12,)


def test_relative_key_ambiguity_is_visible():
    """Every white-key pitch class equally often: C major, with A minor a close second.

    Nothing in an evenly played scale marks a tonic, so the relative minor (same
    pitch classes) is the runner-up and the margin is small. The cadences above
    avoid this by sounding the tonic chord twice and the minor's leading tone.
    """
    import denote

    n = int(SR * 0.4)
    t = np.arange(n) / SR
    notes = [60, 62, 64, 65, 67, 69, 71] * 2
    y = np.concatenate(
        [
            sum(
                np.sin(2 * np.pi * k * 440 * 2 ** ((m - 69) / 12) * t) / k
                for k in (1, 2, 3)
            )
            for m in notes
        ]
    ).astype(np.float32)
    result = denote.get_key(y, sr=SR)
    assert result.key == "C major"
    assert result.runner_up == result.relative == "A minor"
    assert result.margin < 0.1


def test_profile_and_chroma_seams():
    import denote

    y = render_progression(7, "major")
    calls = []

    def chroma(y, sr):
        calls.append(sr)
        import librosa

        return librosa.feature.chroma_cqt(y=y, sr=sr)

    result = denote.get_key(y, sr=SR, profile="temperley", chroma=chroma)
    assert calls == [SR]
    assert result.profile == "temperley"
    assert result.key == "G major"
    assert denote.get_key(y, sr=SR, chroma="stft").key == "G major"


def test_unknown_chroma_name_raises():
    import denote

    with pytest.raises(ValueError, match="unknown chroma"):
        denote.get_key(render_progression(0, "major"), sr=SR, chroma="nope")


def test_silence_raises():
    import denote

    with pytest.raises(ValueError, match="flat"):
        denote.get_key(np.zeros(SR, dtype=np.float32), sr=SR)


def test_get_key_from_file(tmp_path):
    sf = pytest.importorskip("soundfile")
    import denote

    path = tmp_path / "cadence.wav"
    sf.write(str(path), render_progression(2, "major"), SR)
    assert denote.get_key(str(path)).key == "D major"


def test_registry_and_services():
    import denote

    assert "librosa_key" in denote.list_backends("key")
    assert denote.get_default_backend("key") == "librosa_key"
    handle = denote.services.librosa_key
    assert "key" in handle.info["tasks"]
    result = handle.get_key(render_progression(5, "major"), sr=SR)
    assert result.key == "F major"
