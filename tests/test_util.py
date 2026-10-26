"""Tests for denote.util's chunked overlap-add (chunk_bounds, chunked_apply)."""

import numpy as np
import pytest

from denote.util import chunk_bounds, chunked_apply

CASES = [  # (n, segment, overlap)
    (1000, 300, 60),
    (1000, 300, 0),
    (1000, 300, 200),  # ramps of a chunk collide: 2 * overlap > segment
    (301, 300, 0),  # the tail is one frame past a segment
    (340, 300, 1),
    (300, 300, 50),
    (120, 300, 50),  # shorter than a segment: one chunk
]


@pytest.mark.parametrize("n, segment, overlap", CASES)
def test_chunks_cover_everything_at_full_length(n, segment, overlap):
    bounds = chunk_bounds(n, segment=segment, overlap=overlap)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(e - s == min(segment, n) for s, e in bounds)
    assert all(e1 - s2 >= overlap for (_, e1), (s2, _) in zip(bounds, bounds[1:]))


@pytest.mark.parametrize("n, segment, overlap", CASES)
def test_chunked_apply_reassembles_a_pointwise_fn_exactly(n, segment, overlap):
    x = np.random.default_rng(0).normal(size=(2, n)).astype(np.float32)
    seen = []

    def fn(chunk):
        seen.append(chunk.shape)
        return np.stack([0.5 * chunk, -chunk, chunk**2])  # (3, channels, m)

    out = chunked_apply(fn, x, segment=segment, overlap=overlap)
    assert out.shape == (3, 2, n) and out.dtype == np.float32
    assert np.allclose(out, np.stack([0.5 * x, -x, x**2]), atol=1e-5)
    assert all(shape == (2, min(segment, n)) for shape in seen)


def test_chunked_apply_crossfades_where_chunk_outputs_disagree():
    n, segment, overlap = 10, 6, 2  # chunks (0, 6), (4, 10): overlap at 4, 5
    calls = iter([0.0, 1.0])
    out = chunked_apply(
        lambda c: np.full(c.shape, next(calls)),
        np.zeros((1, n)),
        segment=segment,
        overlap=overlap,
    )
    # linear ramps 1/3, 2/3 over the two overlapping frames, no jump
    assert np.allclose(out[0], [0, 0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1])


@pytest.mark.parametrize(
    "segment, overlap", [(300, 300), (300, 400), (300, -1), (0, 0)]
)
def test_chunked_apply_rejects_bad_overlaps(segment, overlap):
    with pytest.raises(ValueError, match="overlap"):
        chunked_apply(
            lambda c: c, np.zeros((1, 1000)), segment=segment, overlap=overlap
        )


def test_chunked_apply_rejects_an_fn_that_changes_the_length():
    with pytest.raises(ValueError, match="frames"):
        chunked_apply(
            lambda c: c[..., :-1], np.zeros((1, 1000)), segment=300, overlap=10
        )
