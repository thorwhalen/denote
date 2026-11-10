"""RoFormer worker: audio-separator's RoFormer checkpoints, run in the ``roformer`` environment.

Loaded standalone by ``ps.isolated.Worker`` (it imports nothing from denote). The
mix arrives as a ``(channels, frames)`` float32 ``.npy`` file at 44.1 kHz; it is
written as a 32-bit float WAV (audio-separator reads files), separated, and each stem
read back and written as ``<stem>.npy`` (lower-case stem name) in the request's folder.
Output WAVs are float too (audio-separator keeps the input's subtype), so no level is
lost to 16-bit quantization; audio-separator scales a stem down only if its peak
exceeds ``normalization_threshold`` (1.0 here, its maximum).
"""

import re

#: audio-separator names outputs ``<input>_(<Stem>)_<model>.<ext>``.
STEM_IN_FILENAME = re.compile(r"_\(([^)]+)\)_")
#: The input's file name in the worker's scratch folder (its stem prefixes outputs).
MIX_WAV = "mix.wav"
#: The rate audio-separator resamples to and writes at.
SAMPLE_RATE = 44100
#: Peak above which audio-separator scales a stem down (1.0 is its largest allowed).
NORMALIZATION_THRESHOLD = 1.0


def load(*, model_file_dir, log_level="WARNING"):
    """The worker's state: audio-separator's settings and its scratch folder."""
    import tempfile

    return {
        "model_file_dir": model_file_dir,
        "log_level": log_level,
        "scratch": tempfile.mkdtemp(prefix="denote-roformer-"),
        "model": None,
        "separator": None,
    }


def _separator(state, model):
    """The separator with checkpoint ``model`` loaded (downloaded on first use)."""
    import logging

    from audio_separator.separator import Separator

    if state["model"] != model:
        state["separator"] = None
        separator = Separator(
            log_level=getattr(logging, state["log_level"]),
            model_file_dir=state["model_file_dir"],
            output_dir=state["scratch"],
            output_format="WAV",
            use_soundfile=True,
            sample_rate=SAMPLE_RATE,
            normalization_threshold=NORMALIZATION_THRESHOLD,
        )
        separator.load_model(model_filename=model)
        state["separator"], state["model"] = separator, model
    return state["separator"]


def stem_name(path):
    """The lower-cased stem name in an audio-separator output file name."""
    import os

    match = STEM_IN_FILENAME.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"no '_(<stem>)_' in the output file name {path!r}")
    return match.group(1).lower()


def separate(state, *, mix, out_dir, sr, model, sources=None):
    """Separate the mix in ``mix`` (a ``.npy`` path); write ``<stem>.npy`` into ``out_dir``.

    Returns:
        ``{"stems": {name: path}, "sr": rate, "sources": the model's stems, "model": model}``; only
        ``sources`` (default: all) are kept.
    """
    import os

    import numpy as np
    import soundfile

    if sr != SAMPLE_RATE:
        raise ValueError(f"the mix must be at {SAMPLE_RATE} Hz, not {sr}")
    separator = _separator(state, model)
    scratch = state["scratch"]
    wav = os.path.join(scratch, MIX_WAV)
    soundfile.write(wav, np.load(mix).T, sr, subtype="FLOAT")
    outputs, rate = [], sr
    try:
        outputs = [
            p if os.path.isabs(p) else os.path.join(scratch, p)
            for p in separator.separate(wav)
        ]
        found = {stem_name(p): p for p in outputs}
        keep = list(found) if sources is None else list(sources)
        if unknown := set(keep) - set(found):
            raise ValueError(
                f"unknown sources {sorted(unknown)}; {model} has {list(found)}"
            )
        paths = {}
        for name in keep:
            audio, rate = soundfile.read(found[name], dtype="float32", always_2d=True)
            path = os.path.join(out_dir, f"{name}.npy")
            np.save(path, np.ascontiguousarray(audio.T))
            paths[name] = path
    finally:
        for p in [wav, *outputs]:  # the scratch folder is reused by the next call
            if os.path.exists(p):
                os.remove(p)
    return {"stems": paths, "sr": rate, "sources": list(found), "model": model}
