# denote.services

### denote.services *= <ServiceCollection backends=['basic_pitch', 'beat_this', 'librosa_beats', 'librosa_key', 'librosa_pyin', 'pesto', 'swift_f0', 'torchaudio_hdemucs', 'torchcrepe']>*

Lazy mapping of backend names → ServiceHandle instances.

Supports both dict-style and attribute-style access:

```default
services['basic_pitch']
services.basic_pitch
```
