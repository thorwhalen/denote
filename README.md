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
result.midi        # pretty_midi.PrettyMIDI object
result.notes       # list of NoteEvent(start_time, end_time, pitch, velocity)

# Pitch estimation (requires torchcrepe, or falls back to librosa pYIN)
pitch = denote.get_pitch("vocal.wav")
pitch.times        # timestamps in seconds
pitch.frequencies  # Hz (NaN for unvoiced)
pitch.confidence   # voicing confidence

# Beat tracking (uses librosa, no extra install needed)
beats = denote.get_beats("song.wav")
beats.beats        # beat times in seconds
beats.tempo        # BPM estimate
```

## Choose Your Backend

```python
# See what's available
denote.list_backends()            # all registered backends
denote.list_backends('pitch')     # ['librosa_pyin', 'torchcrepe']

# Specify a backend explicitly
pitch = denote.get_pitch("vocal.wav", backend='librosa_pyin')
```

## Service-Level Access

For more control, access backends via the service layer:

```python
# Per-backend service handle
denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.3)
denote.services.torchcrepe.get_pitch("vocal.wav", model='tiny', device='cuda')

# Native adapter (full backend API)
denote.services.basic_pitch.adapter
```

## Supported Backends

| Backend | Task | License | Install |
|---------|------|---------|---------|
| Basic Pitch (Spotify) | MIDI transcription | Apache 2.0 | `pip install denote[basic_pitch]` |
| torchcrepe | Pitch estimation | MIT | `pip install denote[torchcrepe]` |
| librosa pYIN | Pitch estimation | ISC | (included) |
| librosa beats | Beat tracking | ISC | (included) |

More backends coming: pesto-pitch, madmom, autochord, beat-this, demucs.

## Plugin System

Register your own backends:

```python
import denote

denote.register_backend('my_tool', config={
    'name': 'my_tool',
    'tasks': ['transcribe'],
    'pip_install': 'my-tool',
}, adapter=MyAdapter(config))
```
