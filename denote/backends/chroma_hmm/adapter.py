"""Adapter for the model-free chord recognizer in ``denote.chords``."""

import numpy as np

from denote.base import ChordResult
from denote.chords import (
    align_sequence,
    decode_chords,
    default_vocabulary,
    path_to_segments,
    peaks_chroma,
)
from denote.translation import make_kwargs_translator
from denote.util import load_audio


class Adapter:
    """Chords as ``[{start, end, label}]`` in Harte syntax, free, vocabulary-restricted or aligned."""

    def __init__(self, config: dict):
        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])

    def get_chords(self, audio, *, sr=None, **kwargs):
        """Recognize the chords of ``audio``.

        Args:
            audio: File path or numpy array.
            sr: Sample rate (required if audio is an array).
            **kwargs: ``vocabulary`` (restrict the chords to this list), ``sequence`` (an ordered
                chord list the audio is known to follow, repeats allowed: returns when each
                happens, one segment per entry, ``raw['positions']`` giving the entry index;
                excludes ``vocabulary`` and ``causal``), ``qualities`` (default vocabulary's
                qualities: ``("maj", "min")``; add ``"7"``, ``"maj7"``, ``"min7"``...),
                ``causal``, ``dwell_s``, ``kappa``, ``harmonics``, ``gate_db``, ``no_chord_cos``, ``window_s``,
                ``hop_s``, ``audio_sr``. With ``sequence``: ``sections`` (one key per entry; the order
                becomes tolerant: jumps between section starts at ``jump_p``, skipped entries at
                ``skip_p``).

        Returns:
            ChordResult: ``intervals`` (N, 2) seconds, Harte ``labels`` (``N`` = no chord),
            ``segments()`` for ``[{start, end, label}]``; ``raw`` holds the per-frame path,
            states and chroma.
        """
        native = self._translate(**kwargs)
        sequence, vocabulary = native["sequence"], native["vocabulary"]
        if sequence is None and native["sections"] is not None:
            raise ValueError("sections= belongs to sequence=")
        if sequence is not None and (vocabulary is not None or native["causal"]):
            raise ValueError(
                "sequence= aligns a known chord list: it excludes vocabulary= and causal="
            )
        y, rate = load_audio(audio, sr=sr, mono=True, target_sr=native["audio_sr"])
        chroma, level = peaks_chroma(
            y, rate, window_s=native["window_s"], hop_s=native["hop_s"]
        )
        hop_s, duration = native["hop_s"], len(y) / rate
        common = dict(
            hop_s=hop_s,
            kappa=native["kappa"],
            gate_db=native["gate_db"],
            no_chord_cos=native["no_chord_cos"],
            harmonics=native["harmonics"],
        )
        raw = {"chroma": chroma, "level_db": level, "hop_s": hop_s}
        if sequence is not None:
            path, states, positions = align_sequence(
                chroma,
                sequence,
                level,
                sections=native["sections"],
                jump_p=native["jump_p"],
                skip_p=native["skip_p"],
                **common,
            )
            segments = path_to_segments(
                path, states, hop_s=hop_s, duration=duration, split_on=positions
            )
            raw["positions"] = [positions[i] for i in _run_starts(positions)]
        else:
            if vocabulary is None:
                vocabulary = default_vocabulary(native["qualities"])
            path, states = decode_chords(
                chroma,
                level,
                vocabulary=vocabulary,
                dwell_s=native["dwell_s"],
                causal=native["causal"],
                **common,
            )
            segments = path_to_segments(path, states, hop_s=hop_s, duration=duration)
        raw.update(path=path, states=states)
        return ChordResult.from_segments(
            segments, raw=raw, backend=self._config["name"]
        )


def _run_starts(key):
    key = list(key)
    return [i for i in range(len(key)) if i == 0 or key[i] != key[i - 1]]
