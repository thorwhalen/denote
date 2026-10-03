# denote

Denote — Portal and facade for audio-to-symbol tools.

Simple usage:

```default
import denote
result = denote.transcribe("song.wav")         # audio -> MIDI
chords = denote.get_chords("song.wav")         # audio -> chord labels
pitch = denote.get_pitch("vocal.wav")           # audio -> F0
beats = denote.get_beats("song.wav")            # audio -> beat times
sync = denote.align_score("song.mid", "song.wav")  # score <-> recording time map

denote.list_backends()                          # see what's available
denote.list_backends('pitch')                   # backends for a task
```

Service-level access:

```default
denote.services.basic_pitch.transcribe("song.wav", onset_threshold=0.3)
denote.services.torchcrepe.get_pitch("vocal.wav", model='tiny')
```

Native adapter access:

```default
denote.services.basic_pitch.adapter  # raw adapter instance
```

### Functions

| [`get_beats`](#denote.get_beats)(audio, \*[, sr, backend])   | Track beats in audio.         |
|----------------------------------------------------------------------------------------|-------------------------------|
| [`get_chords`](#denote.get_chords)(audio, \*[, sr, backend])  | Recognize chords from audio.  |
| [`get_pitch`](#denote.get_pitch)(audio, \*[, sr, backend])   | Estimate pitch/F0 from audio. |
| [`transcribe`](#denote.transcribe)(audio, \*[, sr, backend])  | Transcribe audio to MIDI.     |

### denote.get_beats(audio, , sr=None, backend=None, \*\*kwargs)

Track beats in audio.

* **Parameters:**
  * **audio** – File path (str/Path) or numpy array.
  * **sr** – Sample rate (required when audio is an array).
  * **backend** – Backend name. Defaults to ‘librosa_beats’.
  * **\*\*kwargs** – Backend-specific parameters (hop_length, start_bpm).
* **Returns:**
  BeatResult with .beats, .downbeats, .tempo fields.

### denote.get_chords(audio, , sr=None, backend=None, \*\*kwargs)

Recognize chords from audio.

* **Parameters:**
  * **audio** – File path (str/Path) or numpy array.
  * **sr** – Sample rate (required when audio is an array).
  * **backend** – Backend name.
  * **\*\*kwargs** – Backend-specific parameters.
* **Returns:**
  ChordResult with .intervals, .labels fields.

### denote.get_pitch(audio, , sr=None, backend=None, \*\*kwargs)

Estimate pitch/F0 from audio.

* **Parameters:**
  * **audio** – File path (str/Path) or numpy array.
  * **sr** – Sample rate (required when audio is an array).
  * **backend** – Backend name. Defaults to ‘torchcrepe’ if installed,
    falls back to ‘librosa_pyin’.
  * **\*\*kwargs** – Backend-specific parameters (min_frequency, max_frequency,
    model, device, hop_length).
* **Returns:**
  PitchResult with .times, .frequencies, .confidence fields.

### denote.transcribe(audio, , sr=None, backend=None, \*\*kwargs)

Transcribe audio to MIDI.

* **Parameters:**
  * **audio** – File path (str/Path) or numpy array.
  * **sr** – Sample rate (required when audio is an array).
  * **backend** – Backend name. Defaults to ‘basic_pitch’ if installed.
  * **\*\*kwargs** – Backend-specific parameters (onset_threshold, min_note_length, etc.)
* **Returns:**
  TranscriptionResult with .midi, .notes, and .raw fields.

### Modules

| [`align`](denote.align.md#module-denote.align)             | Align a symbolic score to a recording of the same music (audio-to-score sync).   |
|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`backends`](denote.backends.md#module-denote.backends)       | Backend packages for denote.                                                     |
| [`base`](denote.base.md#module-denote.base)               | Core types and result dataclasses for denote.                                    |
| [`registry`](denote.registry.md#module-denote.registry)       | Backend discovery, registration, and lazy loading.                               |
| [`services`](denote.services.md#denote.services)              | Lazy mapping of backend names → ServiceHandle instances.                         |
| [`translation`](denote.translation.md#module-denote.translation) | Parameter translation between normalized and native backend interfaces.          |
| [`util`](denote.util.md#module-denote.util)               | Audio loading and utility functions for denote.                                  |
