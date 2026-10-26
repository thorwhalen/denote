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
    auto = denote.align_score(
        score, up_a_tone, sr=SR, subsequence=False, transpose="auto"
    )
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


def test_stereo_frames_by_channels_array():
    score = _score(0.5)
    mono = _audio(_score(0.55))
    stereo = np.stack([mono, mono], axis=1)  # (N, 2), soundfile's layout
    a = denote.align_score(score, stereo, sr=SR, subsequence=False)
    assert a.cost < 0.2


def test_impossible_tempo_ratio_raises_a_clear_error():
    ref = _audio(_score(0.5))
    much_longer = _audio(_score(1.5))  # 3x slower, outside [1/2, 2]
    with pytest.raises(ValueError, match="allowed tempo range"):
        denote.align_audio(much_longer, ref, sr=SR, reference_sr=SR)


def test_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        denote.align_score(_score(0.5), np.zeros(1000, dtype="float32"), sr=SR)


def _chroma_of(pitches, frames_per_note):
    """A unit chroma sequence holding each pitch class for ``frames_per_note`` frames."""
    cols = []
    for p in pitches:
        col = np.zeros(12)
        col[p % 12] = 1.0
        cols += [col] * frames_per_note
    return np.array(cols).T


def test_align_chroma_on_precomputed_features():
    reference = _chroma_of(_MELODY, 5)  # 10 fps: 0.5 s a note
    query = _chroma_of(_MELODY[4:12], 6)  # notes 4-11, 20% slower
    a = denote.align_chroma(query, reference, frame_rate=10)
    assert a.cost == pytest.approx(0.0, abs=0.05)
    # note 4 starts at 2.0 s in the reference and at 0 in the query
    assert a.audio_to_score(0.0) == pytest.approx(2.0, abs=0.2)
    assert a.audio_to_score(4.8) == pytest.approx(
        6.0, abs=0.3
    )  # where note 12 would start


def test_align_chroma_transpose_auto_and_scale():
    reference = _chroma_of(_MELODY, 5)
    shifted = np.roll(reference, 2, axis=0)  # the same tune two semitones up
    a = denote.align_chroma(shifted, reference, subsequence=False, transpose="auto")
    assert a.transpose == 2 and a.cost == pytest.approx(0.0, abs=1e-6)
    b = denote.align_chroma(3 * reference, reference, subsequence=False)
    assert b.cost == pytest.approx(0.0, abs=1e-6)  # cosine: scale is irrelevant
    assert np.array_equal(b.query_times, b.audio_times)
    assert np.array_equal(b.reference_times, b.score_times)


def test_align_chroma_longer_query_places_the_reference_inside_it():
    reference = _chroma_of(_MELODY[4:12], 5)  # 8 notes, 4 s
    query = _chroma_of(_MELODY, 5)  # all 16 notes, 8 s: the reference is inside it
    a = denote.align_chroma(query, reference, frame_rate=10)
    assert a.cost == pytest.approx(0.0, abs=0.05)
    assert a.query_times[0] == pytest.approx(2.0, abs=0.2)  # note 4 of the query
    assert a.query_times[-1] == pytest.approx(5.9, abs=0.3)
    assert a.raw["path"][:, 0].max() < query.shape[1]  # (query, reference) pairs
    assert a.raw["path"][:, 1].max() < reference.shape[1]


def test_align_chroma_steps_are_query_reference_whichever_is_longer():
    # steps that let the query advance 2 or 3 frames per reference frame, never fewer:
    # a query 2.5x as slow as the reference must use them, whichever side is longer
    steps = np.array([[1, 1], [2, 1], [3, 1]])
    reference = _chroma_of(_MELODY[:8], 2)
    slow = _chroma_of(_MELODY[:8], 5)  # 2.5x the reference's frames
    a = denote.align_chroma(slow, reference, steps=steps, frame_rate=10)
    assert a.cost == pytest.approx(0.0, abs=0.05)
    # the same steps the other way round cannot fit a slower query
    with pytest.raises(ValueError, match="with these steps"):
        denote.align_chroma(reference, slow, subsequence=False, steps=steps)


def test_align_chroma_silent_frames():
    reference = _chroma_of(_MELODY, 5)
    query = reference.copy()
    query[:, 20:30] = 0  # two notes of silence
    uniform = denote.align_chroma(query, reference, subsequence=False)
    zero = denote.align_chroma(query, reference, subsequence=False, silent="zero")
    # silence is 1 - 1/sqrt(12) from a note as uniform, 1 (nothing alike) as zero
    assert zero.cost > uniform.cost > 0
    with pytest.raises(ValueError, match="silent"):
        denote.align_chroma(query, reference, silent="nan")


@pytest.mark.parametrize(
    "query, match",
    [
        (np.full((12, 20), np.nan), "NaN"),
        (np.ones((24, 20)), "rows"),
        (np.ones(20), "2-D"),
        (np.ones((12, 1)), "too short"),
    ],
)
def test_align_chroma_names_bad_input(query, match):
    with pytest.raises(ValueError, match=match):
        denote.align_chroma(query, _chroma_of(_MELODY, 2))


def test_align_chroma_transpose_needs_twelve_rows():
    with pytest.raises(ValueError, match="12-row"):
        denote.align_chroma(np.ones((6, 10)), np.ones((6, 10)), transpose="auto")
