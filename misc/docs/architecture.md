# Denote Architecture

## Overview

**Denote** is a portal and facade for audio-to-symbol tools. It provides:

1. **Facade**: Normalized interfaces for MIDI transcription, chord recognition,
   pitch estimation, beat tracking, and source separation.
2. **Portal**: Discovery, installation guidance, and simplified usage of 10+
   backend tools via a plugin/registry system.
3. **Plugin System**: A registration mechanism allowing third parties to add
   custom backends.

## Design Principles

- **Progressive disclosure**: Simple one-liner for common tasks; full control
  available via service handles and native APIs.
- **Optional dependencies**: Only `librosa` and `numpy` are required. Each backend
  is an optional extra (`pip install denote[basic_pitch]`).
- **Task-first organization**: The facade is organized by task (transcribe, chords,
  pitch, beats), not by backend.
- **Lazy loading**: Backends are discovered at import time but only instantiated
  on first use.
- **Permissive core**: Core library is MIT. GPL/AGPL backends are optional extras
  with lazy imports.

## Module Structure

```
denote/
├── __init__.py              # Facade: transcribe(), get_chords(), get_pitch(), get_beats()
├── base.py                  # Result dataclasses, AudioInput type, task enums
├── registry.py              # Backend discovery, registration, lazy loading
├── services.py              # ServiceCollection, ServiceHandle, SliceMapping
├── translation.py           # Parameter normalization and validation
├── util.py                  # Audio loading, format conversion helpers
├── backends/
│   ├── __init__.py
│   ├── _base.py             # BaseBackend protocol / ABC
│   ├── basic_pitch/
│   │   ├── __init__.py
│   │   ├── config.py        # BACKEND_CONFIG dict
│   │   ├── adapter.py       # BasicPitchAdapter
│   │   └── AI_CONTEXT.md    # Backend-specific docs for AI agents
│   ├── torchcrepe/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── adapter.py
│   │   └── AI_CONTEXT.md
│   ├── madmom_chords/
│   │   ├── ...
│   ├── librosa_pyin/
│   │   ├── ...
│   ├── librosa_beats/
│   │   ├── ...
│   ├── pesto/
│   │   ├── ...
│   ├── autochord/
│   │   ├── ...
│   ├── beat_this/
│   │   ├── ...
│   └── demucs/
│       ├── ...
└── tests/
    ├── __init__.py
    ├── test_base.py
    ├── test_registry.py
    ├── test_services.py
    ├── test_translation.py
    └── backends/
        ├── test_basic_pitch.py
        ├── test_torchcrepe.py
        └── ...
```

## Core Types (base.py)

```python
from dataclasses import dataclass, field
from typing import Union, Optional, Any, List
from pathlib import Path
import numpy as np

# Input type that all facades accept
AudioInput = Union[str, Path, np.ndarray, 'torch.Tensor']

@dataclass
class NoteEvent:
    start_time: float       # seconds
    end_time: float         # seconds
    pitch: int              # MIDI note number
    velocity: float         # 0.0 - 1.0
    pitch_bends: Optional[List[int]] = None

@dataclass
class TranscriptionResult:
    midi: Any               # pretty_midi.PrettyMIDI (typed as Any to avoid hard dep)
    notes: List[NoteEvent]
    raw: Any = None         # backend-specific raw output
    backend: str = ''

@dataclass
class PitchResult:
    times: np.ndarray       # seconds
    frequencies: np.ndarray # Hz (NaN for unvoiced)
    confidence: np.ndarray  # [0, 1]
    raw: Any = None
    backend: str = ''

@dataclass
class ChordResult:
    intervals: np.ndarray   # (N, 2) array of [start, end] times
    labels: List[str]       # chord symbol strings
    raw: Any = None
    backend: str = ''

@dataclass
class BeatResult:
    beats: np.ndarray       # beat times in seconds
    downbeats: np.ndarray   # downbeat times (may be empty)
    tempo: Optional[float] = None
    raw: Any = None
    backend: str = ''

class Task:
    """Enum-like for supported task types."""
    TRANSCRIBE = 'transcribe'
    CHORDS = 'chords'
    PITCH = 'pitch'
    BEATS = 'beats'
    SEPARATE = 'separate'
```

## Registry System (registry.py)

Following arioso's pattern but adapted for task-based organization:

```python
_registry: Dict[str, Dict] = {}  # backend_name -> {config, adapter, tasks}

def discover_backends() -> List[str]:
    """Scan denote.backends.* for BACKEND_CONFIG dicts."""

def register_backend(name: str, config: dict, adapter=None):
    """Register a third-party backend. Plugin entry point."""

def get_backend(name: str) -> dict:
    """Get backend config + lazily-loaded adapter."""

def list_backends(task: Optional[str] = None) -> List[str]:
    """List available backends, optionally filtered by task."""

def get_default_backend(task: str) -> str:
    """Return the default backend for a given task."""
```

## Backend Config Schema

Each backend provides a `BACKEND_CONFIG` dict:

```python
BACKEND_CONFIG = {
    'name': 'basic_pitch',
    'display_name': 'Basic Pitch (Spotify)',
    'pip_install': 'basic-pitch>=0.3',
    'import_name': 'basic_pitch',
    'license': 'Apache-2.0',
    'tasks': ['transcribe'],
    'default_for': ['transcribe'],       # this is the default for transcription
    'supports_realtime': False,
    'param_map': {
        'audio': {'native_name': 'audio_path'},
        'onset_threshold': {'native_name': 'onset_threshold', 'default': 0.5},
        'frame_threshold': {'native_name': 'frame_threshold', 'default': 0.3},
        'min_note_length': {
            'native_name': 'minimum_note_length',
            'coerce': lambda x: x * 1000,    # seconds -> ms
            'default': 0.128,
        },
        'min_frequency': {'native_name': 'minimum_frequency'},
        'max_frequency': {'native_name': 'maximum_frequency'},
        'device': None,                       # not supported
    },
}
```

## Service Layer (services.py)

Three access levels, following arioso:

```python
# Level 1: Simple facade (denote/__init__.py)
import denote
result = denote.transcribe("song.wav")
chords = denote.get_chords("song.wav")
pitch = denote.get_pitch("vocal.wav")
beats = denote.get_beats("song.wav")

# Level 2: Backend-specific via service handles
result = denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.3)
pitch = denote.services.torchcrepe.get_pitch("vocal.wav", model='tiny')

# Level 3: Native API passthrough
adapter = denote.services.basic_pitch.adapter
model_output, midi, notes = adapter.predict("song.wav")
```

## Audio Loading (util.py)

Centralized audio loading that normalizes all input types:

```python
def load_audio(
    audio: AudioInput,
    sr: Optional[int] = None,
    mono: bool = True,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from file path, numpy array, or torch tensor.

    Returns (audio_array, sample_rate).
    """
```

## Translation Layer (translation.py)

Per-backend parameter mapping:

```python
def make_kwargs_translator(param_map: dict) -> Callable:
    """Create a function that translates normalized kwargs to native kwargs."""

def validate_params(kwargs: dict, param_map: dict) -> dict:
    """Validate parameter types and ranges."""
```

## Testing Strategy

Tests are **conditional on backend availability**:

```python
import pytest

basic_pitch_available = pytest.importorskip("basic_pitch", reason="basic-pitch not installed")

class TestBasicPitchBackend:
    def test_transcribe_returns_result(self, audio_fixture):
        result = denote.transcribe(audio_fixture, backend='basic_pitch')
        assert isinstance(result, TranscriptionResult)
        assert result.midi is not None
        assert len(result.notes) > 0
```

Core tests (registry, services, translation, base types) run without any backends.

## Plugin System

Third parties can register backends via:

1. **Entry points** (preferred):
```toml
[project.entry-points."denote.backends"]
my_backend = "my_package.denote_plugin:BACKEND_CONFIG"
```

2. **Programmatic registration**:
```python
import denote
denote.register_backend('my_tool', config=my_config, adapter=MyAdapter)
```

## Dependency Graph

```
denote (core)
├── numpy
├── librosa (audio loading, basic DSP)
└── [optional backends]
    ├── basic-pitch (transcription)
    ├── torchcrepe (pitch)
    ├── pesto-pitch (pitch, lightweight)
    ├── piano-transcription-inference (piano MIDI)
    ├── madmom (chords, beats)
    ├── autochord (chords)
    ├── beat-this (beats)
    ├── demucs (separation)
    └── pretty-midi (MIDI manipulation)
```
