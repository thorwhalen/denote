"""torchcrepe frame times follow its 16 kHz hop, not the requested one (thorwhalen/denote#10).

Pure arithmetic: no torchcrepe needed.
"""

import pytest

from denote.backends.torchcrepe.adapter import frame_period


@pytest.mark.parametrize(
    "hop, sr, frames_at_16k",
    [
        (220, 22050, 159),  # 159.64 truncated: 219.12 samples at 22050, not 220
        (512, 44100, 185),  # 185.76 truncated
        (441, 44100, 160),  # the 10 ms default: exact
    ],
)
def test_frame_period_is_torchcrepes_own_hop(hop, sr, frames_at_16k):
    assert frame_period(hop, sr) == pytest.approx(frames_at_16k / 16000)


def test_frame_period_at_the_model_rate():
    assert frame_period(160, 16000) == pytest.approx(0.01)


def test_frame_period_does_not_drift_at_22050():
    # 5 minutes of 220-sample hops at 22050 Hz: hop / sr would put the last frame 1.2 s late
    n = int(300 * 22050 / 220)
    assert (n - 1) * 220 / 22050 - (n - 1) * frame_period(220, 22050) > 1.0


def test_get_pitch_times_end_with_the_audio():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torchcrepe")
    import denote

    sr = 22050
    y = np.zeros(10 * sr, dtype=np.float32)
    y[6 * sr : 8 * sr] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)
    r = denote.get_pitch(y, sr=sr, backend="torchcrepe", model="tiny")
    assert r.times[-1] == pytest.approx(10.0, abs=0.02)  # hop / sr ended at 10.04
    voiced = r.times[~np.isnan(r.frequencies)]
    assert voiced[0] == pytest.approx(6.0, abs=0.03)
