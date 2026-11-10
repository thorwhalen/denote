# denote.backends.demucs_mlx.config

Configuration for the demucs-mlx (HTDemucs on MLX) separation backend.

### Module Attributes

| [`ENV_LOCK`](#denote.backends.demucs_mlx.config.ENV_LOCK)   | Every package of the environment, pinned (`python -m denote.isolated lock demucs_mlx`).   |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`MODELS`](#denote.backends.demucs_mlx.config.MODELS)     | The models this backend runs, with their stems (in the model's order).                    |

### denote.backends.demucs_mlx.config.ENV_LOCK *= PosixPath('/home/runner/work/denote/denote/denote/backends/demucs_mlx/env.lock')*

Every package of the environment, pinned (`python -m denote.isolated lock demucs_mlx`).

### denote.backends.demucs_mlx.config.MODELS *= {'htdemucs': ('drums', 'bass', 'other', 'vocals'), 'htdemucs_6s': ('drums', 'bass', 'other', 'vocals', 'guitar', 'piano'), 'htdemucs_ft': ('drums', 'bass', 'other', 'vocals')}*

The models this backend runs, with their stems (in the model’s order).
