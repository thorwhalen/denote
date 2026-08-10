# Backend API Comparison for Denote Facade Design

This document compares the actual Python APIs of the backends denote will wrap,
to inform the design of our normalized interface.

---

## 1. MIDI Transcription (audio → MIDI)

### basic_pitch.inference.predict

```python
predict(
    audio_path: Union[Path, str],
    model_or_model_path: Union[Model, Path, str],
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.7,      # ms
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[Path] = None,
    midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]]]
```

**Returns:** `(model_output_dict, midi_data, note_events)`
- `model_output_dict`: raw model activations (onset, note, contour)
- `midi_data`: `pretty_midi.PrettyMIDI` object
- `note_events`: list of `(start_time, end_time, pitch, amplitude, pitch_bends)`

**Key observations:**
- Input is a file path (not an array)
- Model is a positional argument (we should hide this behind a default)
- `minimum_note_length` is in **milliseconds**
- Returns a 3-tuple; the MIDI object is element [1]

### piano_transcription_inference (ByteDance)

```python
# From documentation / source
transcriptor = PianoTranscription(device="cuda", checkpoint_path=None)
transcriptor.transcribe(audio, midi_path)
# audio: np.ndarray (mono, 16kHz expected)
# midi_path: str, path to save MIDI
```

**Key observations:**
- Class-based API (instantiate then call `.transcribe()`)
- Input is a numpy array (not a file path) — opposite of basic_pitch
- Hardcoded to 16kHz sample rate
- Outputs pedal events (CC64) — unique feature

### omnizart.music

```python
# CLI: omnizart music transcribe <audio_path> [--model-path] [--output]
# Python:
from omnizart.music import app as music_app

music_app.transcribe(input_audio, model_path=None, output=None)
```

**Key observations:**
- CLI-first design, Python API is thin wrapper
- Separate modules for different tasks (music, chord, drum, vocal)

---

## 2. Pitch Estimation (audio → F0)

### torchcrepe.predict

```python
torchcrepe.predict(
    audio,                          # torch.Tensor (1, T)
    sample_rate,                    # int
    hop_length=None,                # int, defaults to sample_rate/100 (10ms)
    fmin=50.0,                      # float, Hz
    fmax=2006.0,                    # float, Hz
    model='full',                   # 'full' or 'tiny'
    decoder=torchcrepe.viterbi,     # callable: viterbi, argmax, weighted_argmax
    return_harmonicity=False,
    return_periodicity=False,
    batch_size=None,
    device='cpu',
    pad=True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]
```

**Returns:** pitch (Hz) tensor, optionally + harmonicity/periodicity

### torchcrepe.predict_from_file

Same params as `predict` but takes `audio_file` (path) instead of `audio` tensor.

### librosa.pyin

```python
librosa.pyin(
    y: np.ndarray,
    *,
    fmin: float,                    # REQUIRED, no default
    fmax: float,                    # REQUIRED, no default
    sr: float = 22050,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    n_thresholds: int = 100,
    beta_parameters: Tuple = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: Optional[float] = nan,
    center: bool = True,
    pad_mode: str = 'constant',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Returns:** `(f0, voiced_flag, voiced_probabilities)`

**Key observations:**
- Input is numpy array + sample rate
- `fmin` and `fmax` are REQUIRED (no defaults) — annoying for users
- Returns 3 arrays (f0, voiced, confidence)
- No GPU support (CPU only)

### PESTO (pesto-pitch)

```python
# From docs:
import pesto

timesteps, pitch, confidence, activations = pesto.predict(
    audio,  # torch.Tensor or np.ndarray
    sample_rate,  # int
    step_size=10.0,  # ms
    reduction="argmax",  # 'argmax', 'mean', 'alwa'
    num_chunks=None,
)
```

**Key observations:**
- Extremely lightweight (30K params)
- Returns 4 values: timesteps, pitch (Hz), confidence, raw activations
- step_size in ms (not samples or hop_length)

---

## 3. Chord Recognition (audio → chord labels)

### madmom (chord pipeline)

```python
from madmom.features.chords import (
    CNNChordFeatureProcessor,
    CRFChordRecognitionProcessor,
)

# Pipeline:
feat_proc = CNNChordFeatureProcessor()
chord_proc = CRFChordRecognitionProcessor()
feats = feat_proc(audio_file)
chords = chord_proc(feats)
# chords: list of (start_time, end_time, chord_label)
```

**Key observations:**
- Processor pipeline pattern (composable)
- Input is file path or Signal object
- Output is `(start, end, label)` tuples — already interval-like
- Only 25 classes (12 major + 12 minor + N)

### autochord

```python
import autochord

result = autochord.recognize(audio_path, lab_fn=None)
# Returns: list of (start_time, end_time, chord_label)
```

**Key observations:**
- Simplest API of all chord tools
- Single function call
- Returns same format as madmom

### Chordino (via vamp)

```python
import vamp

result = vamp.collect(
    audio_data,  # np.ndarray
    sample_rate,  # int
    "nnls-chroma:chordino",
    parameters={
        "useNNLS": 1,
        "rollon": 0.0,
        "tuningmode": 0,
        "whitening": 1.0,
        "s": 0.7,  # smoothing
        "boostn": 0.1,
    },
)
# Returns: dict with 'list' key containing (timestamp, duration, label) dicts
```

---

## 4. Beat Tracking (audio → beat times)

### librosa.beat.beat_track

```python
librosa.beat.beat_track(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100,
    trim: bool = True,
    bpm: Optional[float] = None,
    prior: Optional[rv_continuous] = None,
    units: str = 'frames',
    sparse: bool = True,
) -> Tuple[float, np.ndarray]
```

**Returns:** `(tempo_estimate, beat_frames)`

### madmom beat tracking

```python
from madmom.features.beats import (
    RNNBeatProcessor,
    DBNBeatTrackingProcessor,
)

beat_proc = RNNBeatProcessor()
beat_track = DBNBeatTrackingProcessor(fps=100)
activations = beat_proc(audio_file)
beats = beat_track(activations)
# beats: np.ndarray of beat times in seconds
```

### Beat This! (beat_this)

```python
# From docs:
from beat_this.inference import File2Beats

file2beats = File2Beats(checkpoint_path=None, device="cpu", dbn=False)
beats, downbeats = file2beats(audio_path)
# beats: np.ndarray of beat times (seconds)
# downbeats: np.ndarray of downbeat times (seconds)
```

---

## 5. Source Separation (audio → stems)

### demucs

```python
# CLI: demucs <audio_path> --out <output_dir> -n htdemucs
# Python API:
from demucs.api import Separator

separator = Separator(model="htdemucs", device="cpu")
origin, separated = separator.separate_audio_file(path)
# separated: dict of {stem_name: torch.Tensor}
# stem names: 'vocals', 'drums', 'bass', 'other'
```

---

## Normalized Interface Design Implications

### Input Patterns Across Backends

| Backend | Input Type | Sample Rate Handling |
|---------|-----------|---------------------|
| basic_pitch | file path | auto-detects from file |
| piano_transcription | np.ndarray (16kHz) | user must resample |
| torchcrepe | torch.Tensor | explicit `sample_rate` param |
| torchcrepe.predict_from_file | file path | auto-loads |
| librosa.pyin | np.ndarray | explicit `sr` param |
| pesto | torch.Tensor or np.ndarray | explicit `sample_rate` param |
| madmom | file path or Signal | auto-loads |
| autochord | file path | auto-loads |
| beat_this | file path | auto-loads |
| demucs | file path | auto-loads |

**Decision:** The denote facade should accept BOTH:
- `audio: Union[str, Path, np.ndarray, torch.Tensor]`
- `sr: Optional[int] = None` (required when audio is an array)

Each adapter handles the conversion internally.

### Output Patterns Across Backends

| Task | Backend | Output Type |
|------|---------|-------------|
| MIDI | basic_pitch | `(dict, PrettyMIDI, note_events)` |
| MIDI | piano_transcription | writes file (side effect) |
| Pitch | torchcrepe | `torch.Tensor` (Hz) |
| Pitch | librosa.pyin | `(np.ndarray, np.ndarray, np.ndarray)` |
| Pitch | pesto | `(timesteps, pitch, confidence, activations)` |
| Chords | madmom | `list[(start, end, label)]` |
| Chords | autochord | `list[(start, end, label)]` |
| Beats | librosa | `(tempo, beat_frames)` |
| Beats | madmom | `np.ndarray` (seconds) |
| Beats | beat_this | `(beats, downbeats)` |

### Proposed Normalized Parameter Names

| Normalized Name | Type | Description | Backends That Use It |
|----------------|------|-------------|---------------------|
| `audio` | Union[str, Path, ndarray] | Audio input | ALL |
| `sr` | int | Sample rate (when audio is array) | ALL (when needed) |
| `onset_threshold` | float [0-1] | Note onset sensitivity | basic_pitch (0.5) |
| `frame_threshold` | float [0-1] | Frame activation threshold | basic_pitch (0.3) |
| `min_note_length` | float | Minimum note duration (seconds) | basic_pitch (0.1277s) |
| `min_frequency` | float | Minimum F0 in Hz | basic_pitch, torchcrepe (50), pyin (REQUIRED) |
| `max_frequency` | float | Maximum F0 in Hz | basic_pitch, torchcrepe (2006), pyin (REQUIRED) |
| `hop_length` | int | Hop size in samples | torchcrepe, pyin, librosa |
| `step_size` | float | Hop size in ms | pesto (10.0) |
| `model` | str | Model variant | torchcrepe ('full'/'tiny'), pesto |
| `device` | str | Compute device | torchcrepe, piano_trans, beat_this, demucs |
| `vocabulary` | str | Chord vocabulary size | madmom ('major_minor'), chordino (configurable) |
| `backend` | str | Which backend to use | denote facade only |

### Proposed Normalized Result Types

```python
@dataclass
class TranscriptionResult:
    midi: pretty_midi.PrettyMIDI
    notes: list[NoteEvent]  # (start, end, pitch, velocity, pitch_bends)
    raw: Any  # backend-specific raw output


@dataclass
class PitchResult:
    times: np.ndarray  # seconds
    frequencies: np.ndarray  # Hz (NaN for unvoiced)
    confidence: np.ndarray  # [0, 1]
    raw: Any


@dataclass
class ChordResult:
    intervals: np.ndarray  # (N, 2) start/end times
    labels: list[str]  # chord labels
    raw: Any


@dataclass
class BeatResult:
    beats: np.ndarray  # beat times in seconds
    downbeats: np.ndarray  # downbeat times (may be empty)
    tempo: Optional[float]  # BPM estimate
    raw: Any
```
