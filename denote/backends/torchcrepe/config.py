"""Configuration for the torchcrepe backend."""

BACKEND_CONFIG = {
    'name': 'torchcrepe',
    'display_name': 'torchcrepe (CREPE PyTorch)',
    'pip_install': 'torchcrepe>=0.0.20',
    'import_name': 'torchcrepe',
    'license': 'MIT',
    'tasks': ['pitch'],
    'default_for': ['pitch'],
    'supports_realtime': False,
    'description': (
        'GPU-accelerated pitch/F0 estimation using CREPE CNN. '
        'Supports Viterbi decoding for smooth output.'
    ),
    'param_map': {
        'audio': {'native_name': 'audio'},
        'sr': {'native_name': 'sample_rate'},
        'hop_length': {'native_name': 'hop_length'},
        'min_frequency': {'native_name': 'fmin', 'default': 50.0},
        'max_frequency': {'native_name': 'fmax', 'default': 2006.0},
        'model': {
            'native_name': 'model',
            'default': 'full',
            'choices': ['full', 'tiny'],
        },
        'device': {'native_name': 'device', 'default': 'cpu'},
        'batch_size': {'native_name': 'batch_size'},
        # Not supported
        'onset_threshold': None,
        'frame_threshold': None,
        'min_note_length': None,
    },
}
