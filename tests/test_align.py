"""Tests for denote.align: score <-> recording and recording <-> recording sync.

The "recordings" are pretty_midi's own sine synthesis of the score played at
another tempo, so the true time map is known.
"""

import numpy as np
import pytest

pretty_midi = pytest.importorskip("pretty_midi")
pytest.importorskip("librosa")

import denote

SR = 22050
_MELODY = [60, 64, 67, 72, 71, 67, 65, 62, 60, 57, 59, 62, 67, 66, 64, 60]


def _score(note_len=0.5, *, pitches=_MELODY, offset=0.0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for k, p in enumerate(pitches):
        t = offset + k * note_len
        inst.notes.append(pretty_midi.Note(90, p, t, t + note_len * 0.95))
        inst.notes.append(pretty_midi.Note(70, p - 12, t, t + note_len * 0.95))
    pm.instruments.append(inst)
    return pm


def _audio(pm):
    return pm.synthesize(fs=SR).astype("float32")


def test_align_score_recovers_a_tempo_change():
    score = _score(0.5)  # 8 s
    performance = _audio(_score(0.6))  # same notes, 20% slower: 9.6 s
    a = denote.align_score(score, performance, sr=SR, subsequence=False)
    # Note k starts at 0.5k in the score and 0.6k in the performance.
    assert a.audio_to_score(6.0) == pytest.approx(5.0, abs=0.3)
    assert a.score_to_audio(5.0) == pytest.approx(6.0, abs=0.3)
    assert a.cost < 0.2


def test_align_score_finds_a_clip_inside_the_score():
    score = _score(0.5, pitches=[50, 52, 53, 55] * 3 + _MELODY)  # melody from 6 s
    clip = _audio(_score(0.5))  # just the melody
    a = denote.align_score(score, clip, sr=SR)
    assert a.score_span[0] == pytest.approx(6.0, abs=0.4)


def test_transpose_auto_finds_the_key_shift():
    score = _score(0.5)
    up_a_tone = _audio(_score(0.5, pitches=[p + 2 for p in _MELODY]))
    plain = denote.align_score(score, up_a_tone, sr=SR, subsequence=False)
    auto = denote.align_score(score, up_a_tone, sr=SR, subsequence=False, transpose="auto")
    assert auto.transpose == 2
    assert auto.cost < plain.cost


def test_align_audio_scores_fidelity():
    ref = _audio(_score(0.5))
    same_slower = _audio(_score(0.55))
    other = _audio(_score(0.5, pitches=list(reversed(_MELODY))))
    close = denote.align_audio(same_slower, ref, sr=SR, reference_sr=SR)
    far = denote.align_audio(other, ref, sr=SR, reference_sr=SR)
    assert close.cost < far.cost


def test_silence_does_not_break_alignment():
    score = _score(0.5, offset=2.0)  # 2 s of silence first
    a = denote.align_score(score, _audio(_score(0.5)), sr=SR)
    assert np.isfinite(a.cost)
