"""Configuration for the librosa pYIN backend."""

BACKEND_CONFIG = {
    'name': 'librosa_pyin',
    'display_name': 'librosa pYIN',
    'pip_install': 'librosa>=0.10',
    'import_name': 'librosa',
    'license': 'ISC',
    'tasks': ['pitch'],
    'default_for': [],  # torchcrepe is default; pyin is a no-GPU fallback
    'supports_realtime': False,
    'description': (
        'Probabilistic YIN pitch estimation. CPU-only, no deep learning. '
        'Good for monophonic sources when GPU is not available.'
    ),
    'param_map': {
        'audio': {'native_name': 'y'},
        'sr': {'native_name': 'sr', 'default': 22050},
        'min_frequency': {'native_name': 'fmin', 'default': 65.0},
        'max_frequency': {'native_name': 'fmax', 'default': 2093.0},
        'hop_length': {'native_name': 'hop_length'},
        'frame_length': {'native_name': 'frame_length', 'default': 2048},
        # Not supported
        'model': None,
        'device': None,
        'onset_threshold': None,
    },
}
