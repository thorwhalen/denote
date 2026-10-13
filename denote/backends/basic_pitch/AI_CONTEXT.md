# Basic Pitch Backend — AI Context

## What it does
Basic Pitch (by Spotify) is a lightweight CNN for polyphonic note transcription.
It converts audio to MIDI, supporting pitch bends. Instrument-agnostic, runs on CPU.

## Native API
```python
from basic_pitch.inference import predict

model_output, midi_data, note_events = predict(
    audio_path,  # str or Path
    onset_threshold=0.5,  # float [0, 1]
    frame_threshold=0.3,  # float [0, 1]
    minimum_note_length=127.7,  # milliseconds
    minimum_frequency=None,  # Hz
    maximum_frequency=None,  # Hz
    multiple_pitch_bends=False,
    melodia_trick=True,
    midi_tempo=120,
)
# Returns: (dict of model activations, PrettyMIDI, list of note tuples)
```

## Key gotchas
- `minimum_note_length` is in **milliseconds**, not seconds
- Input must be a file path (not an array) — adapter handles conversion
- Returns a 3-tuple; MIDI object is element [1]
- The `model_or_model_path` param defaults to whichever runtime basic-pitch detects; the adapter passes the ONNX model whenever onnxruntime is importable
- Needs a model runtime: onnxruntime (portable, in the `basic_pitch` extra), tflite-runtime, tensorflow or coremltools
- Import it through `compat.import_basic_pitch()`, never directly: on current dependencies basic-pitch breaks on a leftover `tensorflow/` folder, on resampy's `pkg_resources` import (setuptools >= 81) and on `scipy.signal.gaussian` (scipy >= 1.13). See `compat.py` and issue #2.
- On Python 3.12 pip resolves basic-pitch only on macOS, and only 0.3.0 (0.3.1+ pin `tensorflow-macos<2.15.1`, which has no cp312 wheel). On Linux no release resolves (each pins a TensorFlow with no cp312 wheel): install `basic-pitch --no-deps` plus librosa, mir_eval, pretty_midi, `resampy<0.4.3`, scikit-learn, scipy and onnxruntime. That is why the floor stays 0.3, and why requiring >=0.4 would not help (0.4.0 fixes only the scipy call).

## Install
```bash
pip install 'denote[basic_pitch]'   # basic-pitch>=0.3 + onnxruntime
```

## License
Apache-2.0 (permissive)
