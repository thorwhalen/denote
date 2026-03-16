# Denote Roadmap

## Phase 1: Core Infrastructure (v0.0.1)

- [x] Project setup (repo, pyproject.toml, CI)
- [ ] `base.py` — Result dataclasses (TranscriptionResult, PitchResult, ChordResult, BeatResult)
- [ ] `util.py` — Audio loading/normalization
- [ ] `registry.py` — Backend discovery, registration, lazy loading
- [ ] `services.py` — ServiceCollection, ServiceHandle
- [ ] `translation.py` — Parameter normalization
- [ ] `__init__.py` — Top-level facade functions

## Phase 2: First Backends (v0.0.2)

Priority backends (permissive licenses, pip-installable, well-maintained):

- [ ] `basic_pitch` — MIDI transcription (Apache 2.0, Spotify)
- [ ] `torchcrepe` — Pitch estimation (MIT)
- [ ] `librosa_pyin` — Pitch estimation, no GPU needed (ISC)
- [ ] `librosa_beats` — Basic beat tracking (ISC)
- [ ] Tests for each backend (conditional on installation)

## Phase 3: Expand Backends (v0.0.3)

- [ ] `pesto` — Lightweight pitch estimation (MIT, 30K params)
- [ ] `autochord` — Simple chord recognition
- [ ] `madmom_chords` — CNN+CRF chord recognition (BSD/CC-NC)
- [ ] `madmom_beats` — RNN+DBN beat tracking
- [ ] `beat_this` — SOTA beat tracking (MIT)
- [ ] `demucs` — Source separation (MIT)

## Phase 4: Advanced Features (v0.0.4)

- [ ] Pipeline support (separate → transcribe each stem → merge)
- [ ] `piano_transcription` — ByteDance piano MIDI (MIT)
- [ ] Entry-point plugin system for third-party backends
- [ ] CLI interface (`denote transcribe song.wav`)

## Phase 5: Higher-Level Analysis (v0.1.0)

- [ ] Key detection (via Essentia or madmom, optional)
- [ ] Structural segmentation (via allin1)
- [ ] Lead sheet generation pipeline (melody + chords + lyrics)
- [ ] Roman numeral analysis pipeline (chords + key → music21)

## Phase 6: Real-Time Support (v0.2.0)

- [ ] Streaming pitch estimation (PESTO, torchcrepe)
- [ ] Real-time beat tracking (BeatNet)
- [ ] Real-time chord recognition
- [ ] Callback/generator-based API for live audio

## Non-Goals (for now)

- MusicXML output (complex quantization problem, defer to music21)
- Training or fine-tuning models
- Audio editing / manipulation (that's Melodyne's domain)
- GUI / web interface
