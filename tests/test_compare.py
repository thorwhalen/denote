"""Tests for denote.compare_audio: harmony and timbre distances on one alignment (denote#3).

Recordings are pretty_midi syntheses of one melody: sine waves at another tempo
(same sound), square waves (same notes, another sound), and the melody reversed.
"""

import numpy as np
import pytest

pretty_midi = pytest.importorskip("pretty_midi")
pytest.importorskip("librosa")
signal = pytest.importorskip("scipy.signal")

import denote
from denote.compare import METRICS, AlignedPair

SR = 22050
_MELODY = [60, 64, 67, 72, 71, 67, 65, 62, 60, 57, 59, 62, 67, 66, 64, 60]


def _score(note_len=0.5, *, pitches=_MELODY):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for k, p in enumerate(pitches):
        t = k * note_len
        inst.notes.append(pretty_midi.Note(90, p, t, t + note_len * 0.95))
    pm.instruments.append(inst)
    return pm


def _audio(pm, wave=np.sin):
    return pm.synthesize(fs=SR, wave=wave).astype("float32")


REF = _audio(_score(0.5))


def _compare(audio, **kw):
    return denote.compare_audio(audio, REF, sr=SR, reference_sr=SR, **kw)


def test_timbre_separates_what_chroma_cannot():
    same_sound = _compare(_audio(_score(0.55)))
    other_sound = _compare(_audio(_score(0.55), wave=signal.square))
    assert same_sound["timbre"] < other_sound["timbre"]
    # the notes are the same, so the harmony distance hardly moves
    assert abs(same_sound["chroma"] - other_sound["chroma"]) < other_sound["timbre"] - same_sound["timbre"]


def test_chroma_is_the_alignment_cost():
    c = _compare(_audio(_score(0.55)))
    assert c["chroma"] == pytest.approx(c.alignment.cost)
    assert set(c.scores) == {"chroma", "timbre"}


def test_identical_recordings_are_at_zero():
    c = _compare(REF)
    assert c["timbre"] == pytest.approx(0.0, abs=1e-6)
    assert c["chroma"] == pytest.approx(0.0, abs=1e-6)


def test_a_custom_metric_reads_the_same_path():
    seen = {}

    def length_ratio(pair: AlignedPair) -> float:
        seen["path"] = pair.path
        return len(pair.audio) / len(pair.reference)

    c = _compare(_audio(_score(0.55)), metrics={"length": length_ratio, **METRICS})
    assert c["length"] == pytest.approx(1.1, rel=0.02)
    assert np.array_equal(seen["path"], c.alignment.raw["path"])


def test_unknown_metric_names_are_refused():
    with pytest.raises(ValueError, match="unknown metric"):
        _compare(REF, metrics=("chroma", "loudness"))
