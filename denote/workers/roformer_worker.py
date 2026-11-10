"""RoFormer worker: audio-separator's RoFormer checkpoints, run in the ``roformer`` environment.

Loaded standalone by ``ps.isolated.Worker`` (it imports nothing from denote). The
mix arrives as a ``(channels, frames)`` float32 ``.npy`` file at 44.1 kHz; it is
written, next to it, as a 32-bit float WAV (audio-separator reads files), separated,
and each stem read back and written as ``<stem>.npy`` (lower-case stem name) in the
request's folder. Output WAVs are float too (audio-separator keeps the input's
subtype), so no level is lost to 16-bit quantization; audio-separator scales a stem
down only if its peak exceeds ``normalization_threshold`` (1.0 here, its maximum).

Stems come back target first (the checkpoint's primary stem, e.g. ``vocals``), then
the rest in the order audio-separator wrote them (it writes the secondary stem first).
audio-separator writes its outputs into one folder fixed when it is built: a
temporary folder of this worker, emptied after each request and removed at exit.
"""

import re

#: audio-separator names outputs ``<input>_(<Stem>)_<model>.<ext>``.
STEM_IN_FILENAME = re.compile(r"_\(([^)]+)\)_")
#: The input's file name in the request's folder (its stem prefixes outputs).
MIX_WAV = "mix.wav"
#: The rate audio-separator resamples to and writes at.
SAMPLE_RATE = 44100
#: Peak above which audio-separator scales a stem down (1.0 is its largest allowed).
NORMALIZATION_THRESHOLD = 1.0


def load(*, model_file_dir, log_level="WARNING"):
    """The worker's state: audio-separator's settings and its output folder."""
    import atexit
    import shutil
    import tempfile

    outputs = tempfile.mkdtemp(prefix="denote-roformer-")
    atexit.register(shutil.rmtree, outputs, True)
    return {
        "model_file_dir": model_file_dir,
        "log_level": log_level,
        "outputs": outputs,
        "model": None,
        "separator": None,
    }


def _separator(state, model):
    """The separator with checkpoint ``model`` loaded (downloaded on first use).

    The state names a separator only once its checkpoint has loaded: a failed load
    leaves no stale name behind, so the next request tries again.
    """
    import logging

    from audio_separator.separator import Separator

    if state["model"] != model:
        state["model"] = state["separator"] = None  # free the previous checkpoint
        separator = Separator(
            log_level=getattr(logging, state["log_level"]),
            model_file_dir=state["model_file_dir"],
            output_dir=state["outputs"],
            output_format="WAV",
            use_soundfile=True,
            sample_rate=SAMPLE_RATE,
            normalization_threshold=NORMALIZATION_THRESHOLD,
        )
        separator.load_model(model_filename=model)
        state["model"], state["separator"] = model, separator
    return state["separator"]


def stem_name(path):
    """The lower-cased stem name in an audio-separator output file name."""
    import os

    match = STEM_IN_FILENAME.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"no '_(<stem>)_' in the output file name {path!r}")
    return match.group(1).lower()


def target_first(names, primary):
    """``names`` with ``primary`` (case-insensitive) moved to the front, if present.

    >>> target_first(["other", "vocals"], "Vocals")
    ['vocals', 'other']
    """
    primary = (primary or "").lower()
    return sorted(names, key=lambda name: name != primary)


def separate(state, *, mix, out_dir, sr, model, sources=None):
    """Separate the mix in ``mix`` (a ``.npy`` path); write ``<stem>.npy`` into ``out_dir``.

    Returns:
        ``{"stems": {name: path}, "sr": rate, "sources": the checkpoint's stems,
        "model": model}``; only ``sources`` (default: all, target first) are kept,
        in that order.
    """
    import os

    import numpy as np
    import soundfile

    if sr != SAMPLE_RATE:
        raise ValueError(f"the mix must be at {SAMPLE_RATE} Hz, not {sr}")
    separator = _separator(state, model)
    wav = os.path.join(out_dir, MIX_WAV)
    soundfile.write(wav, np.load(mix).T, sr, subtype="FLOAT")
    outputs = []
    try:
        outputs = [
            p if os.path.isabs(p) else os.path.join(state["outputs"], p)
            for p in separator.separate(wav)
        ]
        found = {stem_name(p): p for p in outputs}
        primary = getattr(separator.model_instance, "primary_stem_name", None)
        names = target_first(list(found), primary)
        keep = names if sources is None else list(sources)
        if unknown := set(keep) - set(found):
            raise ValueError(f"unknown sources {sorted(unknown)}; {model} has {names}")
        paths, rate = {}, sr
        for name in keep:
            audio, rate = soundfile.read(found[name], dtype="float32", always_2d=True)
            path = os.path.join(out_dir, f"{name}.npy")
            np.save(path, np.ascontiguousarray(audio.T))
            paths[name] = path
    finally:
        for p in [wav, *outputs]:
            if os.path.exists(p):
                os.remove(p)
    return {"stems": paths, "sr": rate, "sources": names, "model": model}
