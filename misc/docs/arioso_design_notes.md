# Arioso Design Notes for Denote

Notes on design patterns from [arioso](https://github.com/thorwhalen/arioso) that
should inform `denote`'s architecture.

## What Arioso Does

Arioso is a unified facade for AI music **generation** platforms (Suno, MusicGen,
ElevenLabs, Lyria, etc.). It wraps 14+ backends behind a single Python interface.

## Key Design Patterns to Reuse

### 1. Progressive Disclosure (3 access levels)

Arioso exposes three tiers of access:

```python
# Simple: one-liner facade
arioso.generate("jazz ballad", platform="sunoapi")

# Intermediate: per-platform service handle
arioso.services.sunoapi.generate(prompt="jazz", duration=30)

# Advanced: platform-specific methods
arioso.services.sunoapi.upload_cover(path)
```

**For denote:** Same pattern applies:
```python
# Simple
denote.transcribe("song.wav")                        # auto-selects best backend
denote.get_chords("song.wav")

# Intermediate
denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.5)

# Advanced
denote.services.basic_pitch.adapter.predict(audio_array, sr)
```

### 2. Registry + Auto-Discovery

Arioso scans `arioso/platforms/<name>/` directories via `pkgutil.iter_modules`.
Each platform provides a `config.py` with `PLATFORM_CONFIG` dict and an optional
`adapter.py`. Registration is lazy: configs are loaded at discovery time, but
adapters are only instantiated on first use.

**For denote:** Same pattern. Each backend lives in `denote/backends/<name>/`.
The config declares:
- What tasks it supports (midi, chords, pitch, beats, etc.)
- Its pip install name and import name
- Parameter mappings to the normalized interface
- License info

### 3. Affordances (Normalized Parameter Vocabulary)

Arioso defines ~40 "affordance" names (prompt, duration, bpm, key, etc.) that
all platforms map to. Each platform's `config.py` includes a `param_map` that
translates affordance names to native parameter names with optional coercion.

**For denote:** Define a normalized vocabulary per task type:
- **Transcription**: `audio`, `onset_threshold`, `min_note_length`, `min_frequency`,
  `max_frequency`, `instrument_filter`
- **Chord recognition**: `audio`, `vocabulary` (major_minor, sevenths, large),
  `hop_length`, `smooth`
- **Pitch estimation**: `audio`, `model_capacity`, `step_size`, `viterbi`
- **Beat tracking**: `audio`, `fps`

### 4. ServiceCollection / ServiceHandle / SliceMapping

- `ServiceCollection`: Lazy mapping of backend names → `ServiceHandle` objects.
  Implements `collections.abc.Mapping`.
- `ServiceHandle`: Per-backend namespace with `.transcribe()` (normalized),
  `.native_transcribe()` (raw), `.adapter`, `.config`, `.info`.
- `SliceMapping`: Cross-backend view. E.g., `denote.transcribers["basic_pitch"]`.

### 5. Translation Layer

Separate module (`translation.py`) that:
- Maps normalized param names → native names
- Applies type coercion (e.g., seconds → milliseconds)
- Validates constraints (min/max/choices)
- Handles unsupported params (warn/raise/ignore)

### 6. Optional Dependencies

In `pyproject.toml`:
```toml
[project.optional-dependencies]
basic_pitch = ["basic-pitch>=0.3"]
torchcrepe = ["torchcrepe>=0.0.20"]
all = ["basic-pitch", "torchcrepe", ...]
```

Lazy imports throughout. Clear error messages when a backend is missing.

### 7. Config-Driven Adapter Generation

For simple REST-API backends, arioso can auto-generate adapters from config alone
(no `adapter.py` needed). The translation layer + HTTP client do the work.

**For denote:** Most backends are Python libraries, not REST APIs, so we'll need
explicit adapters. But the config-driven approach still applies for parameter
mapping and validation.

## Patterns to Adapt (Not Copy Directly)

### Task-Oriented Instead of Platform-Oriented

Arioso organizes by platform (Suno, MusicGen, etc.) because the task is always
the same (text → audio). Denote needs a **task-first** organization because
backends may serve different tasks:

- `basic_pitch` → MIDI transcription
- `torchcrepe` → pitch estimation
- `madmom` → chords, beats, onsets
- `omnizart` → MIDI, chords, drums, vocals

So the facade should be task-first:
```python
denote.transcribe()      # MIDI from audio
denote.get_chords()      # chords from audio
denote.get_pitch()       # F0 from audio
denote.get_beats()       # beats from audio
```

Each task has a default backend, and the user can override via `backend=`.

### Result Types

Arioso returns `Song` (audio bytes/url + metadata). Denote needs task-specific
result types that are compatible with mir_eval and pretty_midi:

- `TranscriptionResult` → wraps `pretty_midi.PrettyMIDI` + metadata
- `ChordResult` → list of `(start, end, label)` intervals
- `PitchResult` → arrays of `(time, frequency, confidence)`
- `BeatResult` → arrays of beat times + optional downbeats

### Multi-Backend Pipelines

Denote should support pipelines like:
1. Separate sources (Demucs)
2. Transcribe each stem (Basic Pitch / ByteDance Piano)
3. Merge into multi-track MIDI

This is beyond arioso's scope but uses the same registry/adapter pattern.

## What NOT to Copy

- REST API machinery (`_base_adapter.py`, session management, polling) — most
  denote backends are local Python libs.
- Named prompts / YAML ledger — not relevant to transcription.
- Async job submission/polling — most transcription is synchronous.
