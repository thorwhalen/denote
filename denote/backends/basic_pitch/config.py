"""Configuration for the Basic Pitch (Spotify) backend."""

BACKEND_CONFIG = {
    'name': 'basic_pitch',
    'display_name': 'Basic Pitch (Spotify)',
    'pip_install': 'basic-pitch>=0.3',
    'import_name': 'basic_pitch',
    'license': 'Apache-2.0',
    'tasks': ['transcribe'],
    'default_for': ['transcribe'],
    'supports_realtime': False,
    'description': (
        'Lightweight, instrument-agnostic polyphonic note transcription. '
        'Supports pitch bends. Runs on CPU.'
    ),
    'param_map': {
        'audio': {'native_name': 'audio_path'},
        'onset_threshold': {
            'native_name': 'onset_threshold',
            'default': 0.5,
            'min': 0.0,
            'max': 1.0,
        },
        'frame_threshold': {
            'native_name': 'frame_threshold',
            'default': 0.3,
            'min': 0.0,
            'max': 1.0,
        },
        'min_note_length': {
            'native_name': 'minimum_note_length',
            'coerce': lambda x: x * 1000,  # seconds -> ms
            'default': 0.128,
        },
        'min_frequency': {'native_name': 'minimum_frequency'},
        'max_frequency': {'native_name': 'maximum_frequency'},
        'pitch_bends': {
            'native_name': 'multiple_pitch_bends',
            'default': False,
        },
        'midi_tempo': {
            'native_name': 'midi_tempo',
            'default': 120.0,
        },
        # Not supported by this backend
        'device': None,
        'model': None,
    },
}
