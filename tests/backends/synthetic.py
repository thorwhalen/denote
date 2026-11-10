"""Synthetic audio with known answers, and the metrics to score backends on it."""

import numpy as np

SR = 22050


def click_track(*, bpm=120.0, seconds=8.0, beats_per_bar=4, first_beat=0.5, sr=SR):
    """A click track: a low kick on each downbeat, a lighter click on other beats.

    Returns ``(y, sr, beats, downbeats)``, times in seconds.
    """
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(int(0.08 * sr)) / sr
    beats = np.arange(first_beat, seconds - 0.2, 60.0 / bpm)
    rng = np.random.default_rng(0)
    for i, b in enumerate(beats):
        if i % beats_per_bar == 0:
            burst = 0.9 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 40)
            burst += 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 25)
        else:
            burst = 0.45 * np.sin(2 * np.pi * 600 * t) * np.exp(-t * 40)
            burst += 0.2 * rng.normal(size=t.size) * np.exp(-t * 60)
        s = int(b * sr)
        y[s : s + t.size] += burst[: n - s].astype(np.float32)
    return y, sr, beats, beats[::beats_per_bar]


def harmonic_tone(*, freqs=(220.0, 330.0, 440.0), seconds_each=1.0, sr=SR):
    """Held notes with five harmonics. Returns ``(y, sr, f0_at)``, ``f0_at(times)`` in Hz."""
    t = np.arange(int(seconds_each * sr)) / sr
    notes = [
        sum(0.5 / k * np.sin(2 * np.pi * k * f * t) for k in range(1, 6)) for f in freqs
    ]
    y = np.concatenate(notes).astype(np.float32)

    def f0_at(times):
        idx = np.clip(
            (np.asarray(times) // seconds_each).astype(int), 0, len(freqs) - 1
        )
        return np.asarray(freqs)[idx]

    return y, sr, f0_at


def f_measure(estimated, reference, *, tolerance=0.07):
    """Beat-tracking F-measure: one-to-one matches within ``tolerance`` seconds."""
    remaining = list(estimated)
    hits = 0
    for r in reference:
        if remaining:
            i = int(np.argmin([abs(e - r) for e in remaining]))
            if abs(remaining[i] - r) <= tolerance:
                hits += 1
                remaining.pop(i)
    if hits == 0:
        return 0.0
    precision, recall = hits / len(estimated), hits / len(reference)
    return 2 * precision * recall / (precision + recall)


def cents_error(estimated_hz, true_hz):
    """Absolute pitch error in cents."""
    return 1200 * np.abs(np.log2(np.asarray(estimated_hz) / np.asarray(true_hz)))


def away_from_changes(times, *, change_every=1.0, margin=0.05, end=None):
    """Mask of frames at least ``margin`` seconds from a note change (and the end)."""
    times = np.asarray(times)
    phase = np.mod(times, change_every)
    keep = (phase > margin) & (phase < change_every - margin)
    if end is not None:
        keep &= times < end - margin
    return keep
