# denote

Portal and facade for audio-to-symbol tools (MIDI, chords, pitch from audio).

## Project Overview

Denote wraps audio analysis backends (4 shipped today: Basic Pitch, torchcrepe, librosa pyin/beats; more planned - see the roadmap in misc/docs)
behind a unified Python facade. Each backend is an optional dependency; only
`numpy` and `librosa` are required. The architecture follows arioso's pattern:
registry → services → adapters, with task-first organization.

## Tech Stack

Python, librosa, numpy. Optional: basic-pitch, torchcrepe, pesto-pitch, madmom,
autochord, beat-this, demucs, pretty-midi.

## Documentation

See `misc/docs/docs_guide.md` for the full docs index. Key docs:
- `misc/docs/architecture.md` — Module structure, design patterns, type system
- `misc/docs/roadmap.md` — Implementation phases and priorities
- `misc/docs/backend_api_comparison.md` — Native API signatures of all backends
- `misc/docs/arioso_design_notes.md` — Design patterns borrowed from arioso

## Module Map

- `denote/__init__.py` — Top-level facade: `transcribe()`, `get_chords()`, `get_pitch()`, `get_beats()`
- `denote/base.py` — Result dataclasses, AudioInput type, NoteEvent
- `denote/registry.py` — Backend discovery, registration, lazy loading
- `denote/services.py` — ServiceCollection, ServiceHandle (3-tier access)
- `denote/translation.py` — Parameter normalization and validation
- `denote/util.py` — Audio loading, format conversion
- `denote/backends/` — One subpackage per backend (config.py + adapter.py + AI_CONTEXT.md)
- `tests/` — Core tests + per-backend conditional tests

## Commands

```bash
pip install -e ".[dev]"
pip install -e ".[basic_pitch,torchcrepe]"  # install specific backends
pytest
pytest -k "not slow"  # skip integration tests
```

## Conventions

- Facade functions accept `audio: Union[str, Path, np.ndarray]` + optional `sr: int`
- All time values in **seconds** (never frames, ms, or samples)
- Result dataclasses always have a `.raw` field for backend-specific output
- Backends are lazy-imported; never import a backend at module level
- Each backend subpackage must have `config.py` with `BACKEND_CONFIG` dict
- Tests that require a backend use `pytest.importorskip()`

## Architecture Patterns

- **Progressive disclosure**: `denote.transcribe()` → `denote.services.basic_pitch.transcribe()` → `adapter.predict()`
- **Registry**: Auto-discovery via `pkgutil.iter_modules` on `denote.backends`
- **Translation**: `param_map` in config translates normalized → native param names
- **Plugin system**: Entry points (`denote.backends`) for third-party registration
