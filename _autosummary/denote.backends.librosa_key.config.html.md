# denote.backends.librosa_key.config

Configuration for the librosa key estimation backend (the single source of its defaults).

### Module Attributes

| [`DFLT_MIN_RMS`](#denote.backends.librosa_key.config.DFLT_MIN_RMS)        | Audio quieter than this overall RMS (-60 dBFS) is refused as "no music to key".   |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`DFLT_MIN_DURATION_S`](#denote.backends.librosa_key.config.DFLT_MIN_DURATION_S) | too few notes to imply a key.                                                     |

### denote.backends.librosa_key.config.DFLT_MIN_DURATION_S *= 1.0*

too few notes to imply a key.

* **Type:**
  Audio shorter than this (seconds) is refused

### denote.backends.librosa_key.config.DFLT_MIN_RMS *= 0.001*

Audio quieter than this overall RMS (-60 dBFS) is refused as “no music to key”.
