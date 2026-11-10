"""demucs-mlx worker: HTDemucs on Apple's MLX, run in the ``demucs_mlx`` environment.

Loaded standalone by ``ps.isolated.Worker`` (it imports nothing from denote). The
mix arrives as a ``(channels, frames)`` float32 ``.npy`` file at the model's rate
(44.1 kHz); each stem is written back as ``<stem>.npy`` in the request's folder.

Normalization is Demucs's own (``demucs.separate``): the mix minus the mean of its
mono reference, divided by that reference's std, and undone on the way out; the
mean comes back as ``mean / n_sources`` per stem, so the stems add up to the mix.
Weights download on first use into demucs-mlx's cache (``~/.cache/demucs-mlx``, or
``$DEMUCS_MLX_CACHE_DIR``), verified against digests shipped with the package.
"""

#: Added to the reference's std, so a silent mix does not divide by zero.
NORM_EPS = 1e-8
#: The model that has one specialist sub-model per stem (single-stem runs are faster).
SINGLE_STEM_MODEL = "htdemucs_ft"


def load(**_):
    """The worker's state: the separator in use, built on first request.

    demucs-mlx downloads weights with urllib, which verifies TLS against the
    interpreter's OpenSSL defaults; an interpreter built without a CA bundle (some
    system and MacPorts Pythons) fails there. certifi's bundle is used unless
    ``SSL_CERT_FILE`` already names one.
    """
    import os

    try:
        import certifi
    except ImportError:
        pass
    else:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    return {"key": None, "separator": None}


def _separator(state, model, stem):
    """The cached separator for ``(model, stem)``, replacing (and freeing) any other."""
    from demucs_mlx import Separator

    key = (model, stem)
    if state["key"] != key:
        state["separator"] = None  # free the previous model's weights first
        state["separator"] = Separator(model=model, stem=stem)
        state["key"] = key
    return state["separator"]


def separate(
    state,
    *,
    mix,
    out_dir,
    sr,
    model="htdemucs",
    sources=None,
    shifts=1,
    overlap=0.25,
    segment=None,
    seed=None,
):
    """Separate the mix in ``mix`` (a ``.npy`` path); write ``<stem>.npy`` into ``out_dir``.

    Returns:
        ``{"stems": {name: path}, "sr": rate, "sources": the model's stems, "model": model}``; only
        ``sources`` (default: all) are written.
    """
    import os

    import numpy as np

    stem = (
        sources[0]
        if model == SINGLE_STEM_MODEL and sources is not None and len(sources) == 1
        else None
    )
    separator = _separator(state, model, stem)
    if separator.samplerate != sr:
        raise ValueError(
            f"{model} runs at {separator.samplerate} Hz; the mix is at {sr}"
        )
    separator.update_parameter(
        shifts=shifts, overlap=overlap, segment=segment, seed=seed
    )
    names = list(separator.model.sources)
    keep = names if sources is None else list(sources)
    if unknown := set(keep) - set(names):
        raise ValueError(f"unknown sources {sorted(unknown)}; {model} has {names}")

    x = np.load(mix).astype(np.float32, copy=False)
    ref = x.mean(axis=0)
    offset, scale = float(ref.mean()), float(ref.std()) + NORM_EPS
    _, stems = separator.separate_tensor(((x - offset) / scale).astype(np.float32))
    paths = {}
    for name in keep:
        path = os.path.join(out_dir, f"{name}.npy")
        stem_audio = np.asarray(stems[name], dtype=np.float32) * scale + offset / len(
            names
        )
        np.save(path, stem_audio)
        paths[name] = path
    return {
        "stems": paths,
        "sr": separator.samplerate,
        "sources": names,
        "model": model,
    }
