"""Tests for the chord benchmark runner (mir_eval scoring over a labelled set)."""

import numpy as np
import pytest

pytest.importorskip("mir_eval")

from denote.base import ChordResult  # noqa: E402
from denote.benchmark import ChordItem, evaluate_chords, score_chords  # noqa: E402

REF = (np.array([[0.0, 2.0], [2.0, 4.0]]), ["C:maj", "A:min"])


def fixed(segments):
    return lambda audio, **kw: ChordResult.from_segments(segments)


def test_perfect_estimate_scores_one():
    est = [
        {"start": 0, "end": 2, "label": "C:maj"},
        {"start": 2, "end": 4, "label": "A:min"},
    ]
    s = score_chords(REF, est)
    assert s["majmin"] == (4.0, 4.0) and s["segmentation"]["n_estimated"] == 2


def test_partial_credit_by_duration_and_loose_metrics():
    est = [{"start": 0, "end": 4, "label": "C:maj"}]
    s = score_chords(REF, est)
    assert s["majmin"][0] == pytest.approx(2.0)
    # A:min vs C:maj share no root: root credit is the same two seconds
    assert s["root"][0] == pytest.approx(2.0)


def test_pooled_is_duration_weighted_and_items_are_named():
    good = [
        {"start": 0, "end": 2, "label": "C:maj"},
        {"start": 2, "end": 4, "label": "A:min"},
    ]
    items = [
        ChordItem("a", REF, name="one"),
        ChordItem("b", ([[0, 4]], ["C:maj"]), name="two"),
    ]
    results = iter([good, [{"start": 0, "end": 4, "label": "G:maj"}]])
    report = evaluate_chords(
        items, recognizer=lambda audio, **kw: ChordResult.from_segments(next(results))
    )
    assert (
        report["items"]["one"]["majmin"] == 1.0
        and report["items"]["two"]["majmin"] == 0.0
    )
    assert report["pooled"]["majmin"] == pytest.approx(0.5) and report["n_items"] == 2


def test_runs_a_registered_backend_on_synthetic_audio():
    from tests.test_chords import PLAN, SR, render

    t, rows = 0.0, []
    for label, dur in PLAN:
        rows.append({"start": t, "end": t + dur, "label": label})
        t += dur
    report = evaluate_chords([ChordItem(render(PLAN), rows, name="synth", sr=SR)])
    assert report["pooled"]["majmin"] > 0.9
