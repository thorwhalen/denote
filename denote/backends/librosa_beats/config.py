"""Configuration for the librosa beat tracking backend."""

BACKEND_CONFIG = {
    'name': 'librosa_beats',
    'display_name': 'librosa Beat Tracker',
    'pip_install': 'librosa>=0.10',
    'import_name': 'librosa',
    'license': 'ISC',
    'tasks': ['beats'],
    'default_for': ['beats'],
    'supports_realtime': False,
    'description': (
        'Dynamic programming beat tracker from librosa. '
        'No deep learning, works on CPU. Adequate for simple use cases.'
    ),
    'param_map': {
        'audio': {'native_name': 'y'},
        'sr': {'native_name': 'sr', 'default': 22050},
        'hop_length': {'native_name': 'hop_length', 'default': 512},
        'start_bpm': {'native_name': 'start_bpm', 'default': 120.0},
        'tightness': {'native_name': 'tightness', 'default': 100},
        # Not supported
        'model': None,
        'device': None,
    },
}
