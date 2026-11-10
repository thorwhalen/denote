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

- [x] `pesto` — Lightweight pitch estimation (LGPL-3.0 code and weights, outputs unrestricted; about 130K params)
- [x] `swift_f0` — Tiny CNN pitch estimation on CPU (MIT, ONNX model in the wheel)
- [ ] `autochord` — Simple chord recognition
- [ ] `madmom_chords` — CNN+CRF chord recognition (BSD/CC-NC)
- [ ] `madmom_beats` — RNN+DBN beat tracking
- [x] `beat_this` — SOTA beat and downbeat tracking (MIT)
- [x] `torchaudio_hdemucs` — Source separation via torchaudio's Hybrid Demucs bundle (no `demucs` package needed)
- [ ] `demucs` — Source separation with the `demucs` package's newer models (htdemucs)
- [x] `weights_license` + `commercial_use` on every backend, `denote.backend_table()`

## Phase 4: Advanced Features (v0.0.4)

- [ ] Pipeline support (separate → transcribe each stem → merge)
- [ ] `piano_transcription` — ByteDance piano MIDI (MIT)
- [ ] Entry-point plugin system for third-party backends
- [ ] CLI interface (`denote transcribe song.wav`)

## Phase 5: Higher-Level Analysis (v0.1.0)

- [x] Key detection: `get_key` with the `librosa_key` backend (Krumhansl-Schmuckler over mean chroma, Krumhansl-Kessler or Temperley profiles)
- [ ] Key detection with a learned model (e.g. a CNN key classifier) and local (windowed) keys
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
