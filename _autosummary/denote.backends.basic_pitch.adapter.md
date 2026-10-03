# denote.backends.basic_pitch.adapter

Adapter for Basic Pitch (Spotify) — audio to MIDI transcription.

### Classes

| [`Adapter`](#denote.backends.basic_pitch.adapter.Adapter)(config)   | Basic Pitch adapter for polyphonic note transcription.   |
|--------------------------------------------------------------------|----------------------------------------------------------|

### *class* denote.backends.basic_pitch.adapter.Adapter(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Basic Pitch adapter for polyphonic note transcription.

#### transcribe(audio, , sr=None, \*\*kwargs)

Transcribe audio to MIDI using Basic Pitch.

* **Parameters:**
  * **audio** – File path or numpy array.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters (onset_threshold, min_note_length, etc.)
* **Returns:**
  TranscriptionResult with .midi, .notes, and .raw fields.
