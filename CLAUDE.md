# denote

Portal and facade for audio-to-symbol tools (MIDI, chords, pitch from audio).

## Project Overview

Denote wraps audio analysis backends (shipped today: Basic Pitch, torchcrepe, SwiftF0, PESTO, librosa pyin/beats/key, Beat This!, torchaudio Hybrid Demucs, and demucs-mlx and RoFormer (audio-separator) in their own environments; more planned - see the roadmap in misc/docs)
behind a unified Python facade. Each backend is an optional dependency; only
`numpy` and `librosa` are required. The architecture follows arioso's pattern:
registry → services → adapters, with task-first organization.

## Tech Stack

Python, librosa, numpy. Optional: basic-pitch, torchcrepe, swift-f0, pesto-pitch, madmom,
autochord, beat-this, demucs, pretty-midi.

## Documentation

See `misc/docs/docs_guide.md` for the full docs index. Key docs:
- `misc/docs/architecture.md` — Module structure, design patterns, type system
- `misc/docs/roadmap.md` — Implementation phases and priorities
- `misc/docs/backend_api_comparison.md` — Native API signatures of all backends
- `misc/docs/arioso_design_notes.md` — Design patterns borrowed from arioso

## Module Map

- `denote/__init__.py` — Top-level facade: `transcribe()`, `get_chords()`, `get_pitch()`, `get_beats()`, `separate()`, `get_key()`, `backend_table()` (licences per backend, configs only)
- `denote/align.py` — `align_score()` / `align_audio()`: chroma DTW time maps; `align_chroma()` for precomputed features
- `denote/compare.py` — `compare_audio()`: chroma + timbre distances on one alignment, metrics as a registry
- `denote/keys.py` — Key profiles (Krumhansl-Kessler, Temperley), `key_from_chroma()` / `key_scores()` for precomputed chroma, key-name helpers
- `denote/base.py` — Result dataclasses (incl. `KeyResult`), AudioInput type, NoteEvent
- `denote/registry.py` — Backend discovery, registration, lazy loading
- `denote/services.py` — ServiceCollection, ServiceHandle (3-tier access)
- `denote/translation.py` — Parameter normalization and validation
- `denote/util.py` — Audio loading, format conversion
- `denote/isolated.py` — Backends that run in their own Python environment (`ps.isolated`): env location (`DENOTE_DATA_DIR`), `EnvNotBuilt`, `build_env()`, `python -m denote.isolated build <backend>`, `IsolatedSeparator` (host side)
- `denote/workers/` — Worker files (`<backend>_worker.py`) loaded standalone in a backend's environment; they import nothing from denote
- `denote/backends/` — One subpackage per backend (config.py + adapter.py + AI_CONTEXT.md); an isolated backend's config adds `isolated_env` (pinned requirements, Python)
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
- Each backend subpackage must have `config.py` with `BACKEND_CONFIG` dict, including `weights_license` and `commercial_use` (one of `denote.base.COMMERCIAL_USE_VALUES`), with the licence's source URL in a comment; `tests/test_backend_table.py` enforces both fields
- Tests that require a backend use `pytest.importorskip()`; isolated backends' worker files are tested in-process against fake upstream modules, and real-environment tests run only with `DENOTE_TEST_REAL_ENVS=1` and the environment built

## Architecture Patterns

- **Progressive disclosure**: `denote.transcribe()` → `denote.services.basic_pitch.transcribe()` → `adapter.predict()`
- **Registry**: Auto-discovery via `pkgutil.iter_modules` on `denote.backends`
- **Translation**: `param_map` in config translates normalized → native param names
- **Plugin system**: Entry points (`denote.backends`) for third-party registration
