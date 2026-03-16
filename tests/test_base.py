"""Tests for denote.base — result types and core structures."""

import numpy as np

from denote.base import (
    NoteEvent,
    TranscriptionResult,
    PitchResult,
    ChordResult,
    BeatResult,
    TASKS,
)


def test_note_event_creation():
    note = NoteEvent(start_time=0.5, end_time=1.0, pitch=60, velocity=0.8)
    assert note.start_time == 0.5
    assert note.end_time == 1.0
    assert note.pitch == 60
    assert note.velocity == 0.8
    assert note.pitch_bends is None


def test_transcription_result_defaults():
    result = TranscriptionResult(midi=None)
    assert result.midi is None
    assert result.notes == []
    assert result.raw is None
    assert result.backend == ''


def test_pitch_result():
    times = np.array([0.0, 0.01, 0.02])
    freqs = np.array([440.0, 441.0, np.nan])
    conf = np.array([0.9, 0.85, 0.1])
    result = PitchResult(times=times, frequencies=freqs, confidence=conf)
    assert len(result.times) == 3
    assert np.isnan(result.frequencies[2])


def test_chord_result():
    intervals = np.array([[0.0, 2.0], [2.0, 4.0]])
    labels = ['C:maj', 'G:maj']
    result = ChordResult(intervals=intervals, labels=labels, backend='test')
    assert result.intervals.shape == (2, 2)
    assert result.labels[0] == 'C:maj'
    assert result.backend == 'test'


def test_beat_result():
    beats = np.array([0.5, 1.0, 1.5, 2.0])
    result = BeatResult(beats=beats, tempo=120.0)
    assert len(result.beats) == 4
    assert result.tempo == 120.0
    assert len(result.downbeats) == 0


def test_tasks_dict():
    assert 'transcribe' in TASKS
    assert 'pitch' in TASKS
    assert 'chords' in TASKS
    assert 'beats' in TASKS
