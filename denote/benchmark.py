"""Score chord recognition against a labelled set with ``mir_eval``, the same way for any backend.

A labelled set is an iterable of ``(audio, reference)`` pairs (or ``ChordItem``); a reference is
``(intervals, labels)`` or ``[{start, end, label}]``, labels in Harte syntax. Scores are
``mir_eval.chord`` weighted chord symbol recall (``root``, ``majmin``, ``sevenths``, ...) plus
segmentation quality, per item and pooled (duration-weighted over the whole set).

>>> from denote.benchmark import ChordItem, evaluate_chords
>>> report = evaluate_chords(items)                      # doctest: +SKIP
>>> report["pooled"]["majmin"]                           # doctest: +SKIP
0.71

``guitarset_items(root)`` yields GuitarSet's microphone comping takes (needs ``jams``);
``load_jams_chords`` reads any JAMS chord annotation. ``mir_eval`` is imported lazily and comes
with the ``eval`` extra.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: mir_eval.chord comparison functions scored by default, loosest to strictest
DFLT_METRICS = ("root", "majmin", "sevenths")


@dataclass
class ChordItem:
    """One labelled recording: audio (path or array + ``sr``) and its reference chords."""

    audio: Any
    reference: Any  # (intervals, labels) or [{start, end, label}]
    name: str = ""
    sr: Optional[int] = None
    kwargs: dict = field(
        default_factory=dict
    )  # per-item backend kwargs (e.g. a sequence)


def _reference(ref) -> Tuple[np.ndarray, list]:
    if (
        isinstance(ref, (tuple, list))
        and len(ref) == 2
        and not isinstance(ref[0], dict)
    ):
        return np.asarray(ref[0], dtype=float).reshape(-1, 2), list(ref[1])
    return (
        np.array([[s["start"], s["end"]] for s in ref], dtype=float).reshape(-1, 2),
        [s["label"] for s in ref],
    )


def _as_item(item, i: int) -> ChordItem:
    if isinstance(item, ChordItem):
        return item
    audio, reference = item[0], item[1]
    name = str(audio) if isinstance(audio, (str, os.PathLike)) else f"item{i}"
    return ChordItem(audio, reference, name=name)


def score_chords(
    reference,
    estimate,
    *,
    duration: Optional[float] = None,
    metrics: Sequence[str] = DFLT_METRICS,
) -> dict:
    """Score one estimate against one reference. Both are ``(intervals, labels)`` or segment lists.

    Returns ``{metric: (credit_seconds, total_seconds), ..., "segmentation": {...}}``: seconds, not
    ratios, so a set can be pooled by summing. Time outside either annotation counts as ``N``."""
    import mir_eval

    ref_i, ref_l = _reference(reference)
    est_i, est_l = _reference(estimate)
    end = (
        duration
        if duration is not None
        else max(ref_i.max(initial=0), est_i.max(initial=0))
    )
    no = mir_eval.chord.NO_CHORD
    est_i, est_l = mir_eval.util.adjust_intervals(est_i, est_l, 0, end, no, no)
    ref_i, ref_l = mir_eval.util.adjust_intervals(ref_i, ref_l, 0, end, no, no)
    iv, ref_m, est_m = mir_eval.util.merge_labeled_intervals(ref_i, ref_l, est_i, est_l)
    dur = mir_eval.util.intervals_to_durations(iv)
    out: Dict[str, Any] = {}
    for name in metrics:
        comp = getattr(mir_eval.chord, name)(ref_m, est_m)
        out[name] = (float((dur * (comp > 0)).sum()), float((dur * (comp >= 0)).sum()))
    out["segmentation"] = {
        "overseg": float(mir_eval.chord.overseg(ref_i, est_i)),
        "underseg": float(mir_eval.chord.underseg(ref_i, est_i)),
        "n_estimated": len(est_l),
        "n_reference": len(ref_l),
    }
    return out


def evaluate_chords(
    items: Iterable,
    *,
    backend: Optional[str] = None,
    recognizer: Optional[Callable] = None,
    metrics: Sequence[str] = DFLT_METRICS,
    **backend_kwargs,
) -> dict:
    """Run a chord recogniser over a labelled set and score it.

    Args:
        items: ``(audio, reference)`` pairs or ``ChordItem``.
        backend: A chords backend name (default: the registry's default for chords).
        recognizer: Instead of a backend, any ``f(audio, **kwargs) -> ChordResult``.
        metrics: ``mir_eval.chord`` comparison names.
        **backend_kwargs: passed to every call (e.g. ``vocabulary=``, ``causal=True``).

    Returns:
        ``{"items": {name: {metric: ratio, ..., "segmentation": ...}}, "pooled": {metric: ratio},
        "n_items": n, "backend": name}``. Pooled ratios weight each item by its duration.
    """
    if recognizer is None:
        import denote

        def recognizer(audio, **kw):
            return denote.get_chords(audio, backend=backend, **kw)

    per_item, credit, total = {}, {m: 0.0 for m in metrics}, {m: 0.0 for m in metrics}
    for i, raw in enumerate(items):
        item = _as_item(raw, i)
        kwargs = {**backend_kwargs, **item.kwargs}
        if item.sr is not None:
            kwargs["sr"] = item.sr
        estimate = recognizer(item.audio, **kwargs)
        scores = score_chords(item.reference, estimate.segments(), metrics=metrics)
        for m in metrics:
            credit[m] += scores[m][0]
            total[m] += scores[m][1]
        per_item[item.name or f"item{i}"] = {
            **{
                m: (scores[m][0] / scores[m][1] if scores[m][1] else float("nan"))
                for m in metrics
            },
            "segmentation": scores["segmentation"],
        }
    return {
        "items": per_item,
        "pooled": {
            m: (credit[m] / total[m] if total[m] else float("nan")) for m in metrics
        },
        "n_items": len(per_item),
        "backend": backend,
    }


def load_jams_chords(
    path, *, index: int = 0, namespace: str = "chord"
) -> Tuple[np.ndarray, list]:
    """``(intervals, labels)`` of the ``index``-th chord annotation of a JAMS file (needs ``jams``)."""
    import jams

    ann = jams.load(str(path), validate=False).search(namespace=namespace)[index]
    return (
        np.array([[o.time, o.time + o.duration] for o in ann.data]).reshape(-1, 2),
        [o.value for o in ann.data],
    )


def guitarset_items(
    root, *, players: Optional[Iterable[int]] = None, annotation: str = "performed"
):
    """GuitarSet microphone comping takes as ``ChordItem``s.

    ``root`` holds ``audio_mono-mic/*_comp_mic.wav`` and ``ann/*.jams`` (the dataset's layout,
    CC BY 4.0). ``players`` restricts to those player numbers 0-5; ``annotation`` is ``"performed"``
    (what was played) or ``"instructed"`` (the lead sheet).
    """
    index = {"instructed": 0, "performed": 1}[annotation]
    wanted = None if players is None else set(players)
    root = os.path.expanduser(str(root))
    for wav in sorted(
        glob.glob(os.path.join(root, "audio_mono-mic", "*_comp_mic.wav"))
    ):
        tid = os.path.basename(wav)[: -len("_mic.wav")]
        if wanted is not None and int(tid[:2]) not in wanted:
            continue
        jam = os.path.join(root, "ann", tid + ".jams")
        yield ChordItem(wav, load_jams_chords(jam, index=index), name=tid)
