"""Configuration for the chroma_hmm chord backend (the single source of its defaults)."""

from denote.chords import (
    DFLT_JUMP_P,
    DFLT_NO_CHORD_COS,
    DFLT_QUALITIES,
    DFLT_SKIP_P,
)

BACKEND_CONFIG = {
    "name": "chroma_hmm",
    "display_name": "Chroma templates + HMM",
    "pip_install": "numpy",
    "import_name": "numpy",
    "license": "MIT",
    # No learned weights: peak chroma, overtone chord templates and a hidden Markov model
    "weights_license": "none",
    "weights_license_note": "no learned weights (templates and an HMM, in denote.chords)",
    "commercial_use": "yes",
    "tasks": ["chords"],
    "default_for": ["chords"],
    "supports_realtime": True,
    "description": (
        "Chord recognition with no model: log-compressed chroma from spectral peaks, chord "
        "templates with each tone's overtones, and an HMM read offline (Viterbi) or causally "
        "(forward filter). `vocabulary=` restricts the chords; `sequence=` aligns a known chord "
        "list to the audio. Output: Harte labels with start/end times. CPU, no download."
    ),
    "param_map": {
        "audio": {"native_name": "y"},
        # restrict the states to these chords (Harte labels, or symbols chordsym reads)
        "vocabulary": {"native_name": "vocabulary", "default": None},
        # force the decode through this ordered chord list (alignment)
        "sequence": {"native_name": "sequence", "default": None},
        # one key per sequence entry (e.g. its score system): the decode may jump between sections
        "sections": {"native_name": "sections", "default": None},
        "jump_p": {"native_name": "jump_p", "default": DFLT_JUMP_P},
        "skip_p": {"native_name": "skip_p", "default": DFLT_SKIP_P},
        # chord qualities of the default vocabulary (every root x each quality)
        "qualities": {"native_name": "qualities", "default": DFLT_QUALITIES},
        # True: forward filter (decision from past frames only); False: offline Viterbi
        "causal": {"native_name": "causal", "default": False},
        # mean seconds a chord lasts: the HMM's self-transition
        "dwell_s": {"native_name": "dwell_s", "default": 2.0},
        # sharpness of the template match (exp(kappa * cosine))
        "kappa": {"native_name": "kappa", "default": 20.0},
        # weight of each overtone of a chord tone in the templates (0: binary)
        "harmonics": {"native_name": "harmonics", "default": 0.6},
        # a frame this far below the loudest so far is no-chord
        "gate_db": {"native_name": "gate_db", "default": 45.0},
        # cosine a frame must beat to be a chord (None: cosine with flat chroma)
        "no_chord_cos": {"native_name": "no_chord_cos", "default": DFLT_NO_CHORD_COS},
        "window_s": {"native_name": "window_s", "default": 0.186},
        "hop_s": {"native_name": "hop_s", "default": 0.046},
        # audio is resampled to this rate first
        "audio_sr": {"native_name": "audio_sr", "default": 22050},
        # Not supported
        "model": None,
        "device": None,
    },
}
