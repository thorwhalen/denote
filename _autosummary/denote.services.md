# denote.services

### denote.services *= <ServiceCollection backends=['basic_pitch', 'librosa_beats', 'librosa_pyin', 'torchcrepe']>*

Lazy mapping of backend names → ServiceHandle instances.

Supports both dict-style and attribute-style access:

```default
services['basic_pitch']
services.basic_pitch
```
