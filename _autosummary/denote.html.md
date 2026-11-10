# denote

Denote — Portal and facade for audio-to-symbol tools.

Simple usage:

```default
import denote
result = denote.transcribe("song.wav")         # audio -> MIDI
chords = denote.get_chords("song.wav")         # audio -> chord labels
pitch = denote.get_pitch("vocal.wav")           # audio -> F0
beats = denote.get_beats("song.wav")            # audio -> beat times
key = denote.get_key("song.wav")                # audio -> "A minor" + scores
sync = denote.align_score("song.mid", "song.wav")  # score <-> recording time map

denote.list_backends()                          # see what's available
denote.list_backends('pitch')                   # backends for a task
denote.backend_table()                          # tasks + weights licence per backend
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

### Module Attributes

| [`BACKEND_TABLE_FIELDS`](#denote.BACKEND_TABLE_FIELDS)   | The config fields [`backend_table()`](#denote.backend_table) reports, in column order.   |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|

### Functions

| [`backend_table`](#denote.backend_table)([task, fields])        | One row per backend: its tasks, code licence, weights licence and commercial use.   |
|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`get_beats`](#denote.get_beats)(audio, \*[, sr, backend])  | Track beats in audio.                                                               |
| [`get_chords`](#denote.get_chords)(audio, \*[, sr, backend]) | Recognize chords from audio.                                                        |
| [`get_key`](#denote.get_key)(audio, \*[, sr, backend])    | Estimate the key (tonic + mode) of a recording.                                     |
| [`get_pitch`](#denote.get_pitch)(audio, \*[, sr, backend])  | Estimate pitch/F0 from audio.                                                       |
| [`separate`](#denote.separate)(audio, \*[, sr, backend])   | Separate a mix into stems (drums, bass, other, vocals, ...).                        |
| [`transcribe`](#denote.transcribe)(audio, \*[, sr, backend]) | Transcribe audio to MIDI.                                                           |

### denote.BACKEND_TABLE_FIELDS *= ('name', 'tasks', 'license', 'weights_license', 'commercial_use')*

The config fields [`backend_table()`](#denote.backend_table) reports, in column order.

### denote.backend_table(task=None, , fields=('name', 'tasks', 'license', 'weights_license', 'commercial_use'))

One row per backend: its tasks, code licence, weights licence and commercial use.

Reads the backends’ configs only, so no adapter (and no optional dependency)
is loaded. `commercial_use` is one of `COMMERCIAL_USE_VALUES`.

* **Parameters:**
  * **task** – If set, only backends for this task (‘pitch’, ‘beats’, …).
  * **fields** – The config keys each row carries (a missing key reads `None`).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
* **Returns:**
  A list of dicts, one per backend (sorted by name), keyed by `fields`.

Example:

```default
free = [r["name"] for r in denote.backend_table("pitch") if r["commercial_use"] == "yes"]
```

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

### denote.get_key(audio, , sr=None, backend=None, \*\*kwargs)

Estimate the key (tonic + mode) of a recording.

* **Parameters:**
  * **audio** – File path (str/Path) or numpy array.
  * **sr** – Sample rate (required when audio is an array).
  * **backend** – Backend name. Defaults to ‘librosa_key’.
  * **\*\*kwargs** – Backend-specific parameters (`profile`: ‘krumhansl’ default or
    ‘temperley’ or a (major, minor) pair; `chroma`: ‘cqt’ default,
    ‘stft’, ‘cens’ or a callable (y, sr) -> (12, frames)).
* **Returns:**
  KeyResult with .key (“A minor”), .tonic, .mode, .confidence (the winner’s
  profile correlation, not a probability), .margin (lead over the runner-up),
  .scores (all 24 keys, best first) and .relative.

Example:

```default
k = denote.get_key("song.wav")
k.key, k.margin, k.ranked[:2]
```

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

### denote.separate(audio, , sr=None, backend=None, \*\*kwargs)

Separate a mix into stems (drums, bass, other, vocals, …).

* **Parameters:**
  * **audio** – File path (str/Path) or numpy array.
  * **sr** – Sample rate (required when audio is an array).
  * **backend** – Backend name. Defaults to ‘torchaudio_hdemucs’. ‘demucs_mlx’ and
    ‘roformer’ run in their own environment (see `denote.isolated`).
  * **\*\*kwargs** – Backend-specific parameters. All take `sources` (stem names
    to keep, in that order). torchaudio_hdemucs: segment_s, overlap_s,
    device, bundle. demucs_mlx: model (‘htdemucs’, ‘htdemucs_ft’,
    ‘htdemucs_6s’), shifts, overlap, seed, build_env. roformer: model (a
    checkpoint file name), build_env. `build_env=True` builds a missing
    environment instead of raising `denote.isolated.EnvNotBuilt`; the
    worker then keeps the model in memory until
    `denote.services[backend].close()` or exit.
* **Returns:**
  (channels, frames)}), .sr, and
  .vocals (mono).
* **Return type:**
  SeparationResult with .stems ({name

Example:

```default
voice = denote.separate("song.mp3").vocals
```

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

| [`align`](denote.align.html.md#module-denote.align)             | Align a symbolic score to a recording of the same music (audio-to-score sync).                                                      |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| [`backends`](denote.backends.html.md#module-denote.backends)       | Backend packages for denote.                                                                                                        |
| [`base`](denote.base.html.md#module-denote.base)               | Core types and result dataclasses for denote.                                                                                       |
| [`compare`](denote.compare.html.md#module-denote.compare)         | Compare a version of a piece with a reference: harmony and sound, on the same moments (denote#3).                                   |
| [`isolated`](denote.isolated.html.md#module-denote.isolated)       | Backends that run in their own Python environment, through `ps.isolated`.                                                           |
| [`keys`](denote.keys.html.md#module-denote.keys)               | Key estimation from pitch-class profiles (the Krumhansl-Schmuckler method).                                                         |
| [`registry`](denote.registry.html.md#module-denote.registry)       | Backend discovery, registration, and lazy loading.                                                                                  |
| [`services`](denote.services.html.md#denote.services)              | Lazy mapping of backend names → ServiceHandle instances.                                                                            |
| [`translation`](denote.translation.html.md#module-denote.translation) | Parameter translation between normalized and native backend interfaces.                                                             |
| [`util`](denote.util.html.md#module-denote.util)               | Audio loading (any layout to `(channels, frames)`), and chunked overlap-add for models that take bounded lengths (`chunked_apply`). |
| [`workers`](denote.workers.html.md#module-denote.workers)         | Worker files: model code that runs in a backend's own environment (`denote.isolated`).                                              |
