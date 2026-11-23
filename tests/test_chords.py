"""Tests for the model-free chord backend (chroma_hmm) and denote.chords, on synthetic chords."""

import numpy as np
import pytest

import denote
from denote import chords as C

SR = 22050
PLAN = [("C:maj", 3.0), ("G:maj", 3.0), ("A:min", 3.0), ("F:maj", 3.0)]


def render(plan, *, sr=SR):
    """Plucked-ish chords (a tone and three overtones per chord tone, re-struck twice a second)."""
    out = []
    for label, dur in plan:
        _, root, intervals = C.parse_chord_label(label)
        t = np.arange(int(dur * sr)) / sr
        y = np.zeros_like(t)
        for iv in intervals:
            f = 440.0 * 2 ** ((48 + root + iv - 69) / 12)
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)):
                y += amp * np.sin(2 * np.pi * f * h * t)
        out.append(y * np.exp(-1.5 * (t % 0.5)) / 3)
    return np.concatenate(out).astype(np.float32)


@pytest.fixture(scope="module")
def audio():
    return render(PLAN)


def labels_at(result, times):
    seg = result.segments()
    return [next(s["label"] for s in seg if s["start"] <= t < s["end"]) for t in times]


MIDDLES = [1.5, 4.5, 7.5, 10.5]
WANT = [label for label, _ in PLAN]


def test_default_backend_is_registered_and_recognizes_chords(audio):
    assert "chroma_hmm" in denote.registry.list_backends("chords")
    result = denote.get_chords(audio, sr=SR)
    assert result.backend == "chroma_hmm"
    assert labels_at(result, MIDDLES) == WANT
    assert result.intervals.shape == (len(result.labels), 2)
    assert result.intervals[0, 0] == 0.0 and result.intervals[-1, 1] == pytest.approx(
        12.0
    )


def test_causal_reads_the_same_chords(audio):
    assert labels_at(denote.get_chords(audio, sr=SR, causal=True), MIDDLES) == WANT


def test_vocabulary_restricts_the_states(audio):
    result = denote.get_chords(audio, sr=SR, vocabulary=["C", "G", "Am", "F"])
    assert set(result.labels) <= {"C:maj", "G:maj", "A:min", "F:maj", "N"}
    assert labels_at(result, MIDDLES) == WANT


def test_sequence_aligns_in_order_with_repeats(audio):
    result = denote.get_chords(
        audio, sr=SR, sequence=["C:maj", "C:maj", "G:maj", "A:min", "F:maj"]
    )
    assert result.raw["positions"] == [
        0,
        1,
        2,
        3,
        4,
    ]  # equal neighbours stay separate segments
    assert result.labels == ["C:maj", "C:maj", "G:maj", "A:min", "F:maj"]
    assert result.intervals[2, 0] == pytest.approx(3.0, abs=0.3)  # G starts near 3 s
    assert (np.diff(result.intervals[:, 0]) >= 0).all()


def test_sequence_forces_a_chord_the_free_decoder_would_not_pick(audio):
    free = denote.get_chords(audio, sr=SR)
    forced = denote.get_chords(
        audio, sr=SR, sequence=["C:maj", "G:maj", "E:min", "F:maj"]
    )
    assert "E:min" not in free.labels and "E:min" in forced.labels


def test_sequence_excludes_vocabulary_and_causal(audio):
    with pytest.raises(ValueError, match="excludes"):
        denote.get_chords(audio, sr=SR, sequence=["C"], vocabulary=["C"])
    with pytest.raises(ValueError, match="fewer than"):
        denote.get_chords(audio[: SR // 4], sr=SR, sequence=["C", "G", "A:min"] * 20)


def test_silence_is_no_chord():
    quiet = np.zeros(SR * 3, dtype=np.float32)
    quiet[::7] = 1e-6
    assert set(denote.get_chords(quiet, sr=SR).labels) == {"N"}


def test_labels_are_harte_and_symbols_are_read():
    assert C.parse_chord_label("A:min7") == ("A:min7", 9, (0, 3, 7, 10))
    assert C.parse_chord_label("Bb")[0] == "A#:maj"
    with pytest.raises(ValueError):
        C.parse_chord_label("H:maj")


def test_segments_round_trip():
    from denote.base import ChordResult

    segs = [
        {"start": 0.0, "end": 1.0, "label": "C:maj"},
        {"start": 1.0, "end": 2.5, "label": "N"},
    ]
    assert ChordResult.from_segments(segs).segments() == segs


def test_default_vocabulary_is_every_root_and_quality():
    assert len(C.default_vocabulary()) == 24
    assert len(C.default_vocabulary(("maj", "min", "7"))) == 36
