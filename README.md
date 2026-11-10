# denote

Portal and facade for audio-to-symbol tools (MIDI, chords, pitch from audio).

Denote wraps multiple audio analysis backends behind a unified Python interface.
Each backend is an optional dependency — install only what you need.

## Install

```bash
pip install denote                          # core only (librosa)
pip install denote[basic_pitch]             # + MIDI transcription
pip install denote[torchcrepe]              # + GPU pitch estimation
pip install denote[all]                     # everything
```

## Quick Start

```python
import denote

# Audio to MIDI (requires basic-pitch)
result = denote.transcribe("song.wav")
result.midi  # pretty_midi.PrettyMIDI object
result.notes  # list of NoteEvent(start_time, end_time, pitch, velocity)

# Pitch estimation (requires torchcrepe, or falls back to librosa pYIN)
pitch = denote.get_pitch("vocal.wav")
pitch.times  # timestamps in seconds
pitch.frequencies  # Hz (NaN for unvoiced)
pitch.confidence  # voicing confidence

# Beat tracking (uses librosa, no extra install needed)
beats = denote.get_beats("song.wav")
beats.beats  # beat times in seconds
beats.tempo  # BPM estimate

# Source separation (requires torch + torchaudio; weights download once)
stems = denote.separate("song.wav")
stems.stems  # {"drums", "bass", "other", "vocals"}: (2, frames) at stems.sr (44.1 kHz)
stems.vocals  # the voice, mono

# Key estimation (uses librosa, no extra install needed)
key = denote.get_key("song.wav")
key.key  # "A minor"
key.confidence  # the winner's profile correlation (not a probability)
key.margin  # lead over the runner-up: small means ambiguous
key.ranked[:3]  # all 24 keys, best first; key.relative names the relative key
```

## Choose Your Backend

```python
# See what's available
denote.list_backends()  # all registered backends
denote.list_backends("pitch")  # ['librosa_pyin', 'pesto', 'swift_f0', 'torchcrepe']

# Each backend's tasks, code licence, weights licence and commercial use
# ('yes', 'no', 'conditional' or 'unclear'), read from configs: nothing is loaded
denote.backend_table("pitch")

# Specify a backend explicitly
pitch = denote.get_pitch("vocal.wav", backend="librosa_pyin")
```

## Service-Level Access

For more control, access backends via the service layer:

```python
# Per-backend service handle
denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.3)
denote.services.torchcrepe.get_pitch("vocal.wav", model="tiny", device="cuda")

# Native adapter (full backend API)
denote.services.basic_pitch.adapter
```

## Supported Backends

| Backend | Task | Code licence | Weights licence (commercial use) | Install |
|---------|------|--------------|----------------------------------|---------|
| Basic Pitch (Spotify) | MIDI transcription | Apache 2.0 | Apache 2.0 (yes) | `pip install denote[basic_pitch]` |
| torchcrepe | Pitch estimation | MIT | MIT (yes) | `pip install denote[torchcrepe]` |
| SwiftF0 | Pitch estimation | MIT | MIT (yes) | `pip install denote[swift_f0]` |
| PESTO | Pitch estimation | LGPL 3.0 | LGPL 3.0 (conditional) | `pip install denote[pesto]` |
| librosa pYIN | Pitch estimation | ISC | no weights (yes) | (included) |
| Beat This! | Beats + downbeats | MIT | MIT (yes) | `pip install denote[beat_this]` |
| librosa beats | Beat tracking | ISC | no weights (yes) | (included) |
| librosa key (Krumhansl-Schmuckler) | Key estimation | ISC | no weights (yes) | (included) |
| Hybrid Demucs (torchaudio) | Source separation | BSD 2-Clause | unstated (unclear) | `pip install denote[torchaudio_hdemucs]` |

`denote.backend_table()` gives the same columns from the backends' configs, where each licence's source is cited.

More backends coming: madmom, autochord, newer separators.

## Plugin System

Register your own backends:

```python
import denote

denote.register_backend(
    "my_tool",
    config={
        "name": "my_tool",
        "tasks": ["transcribe"],
        "pip_install": "my-tool",
    },
    adapter=MyAdapter(config),
)
```
