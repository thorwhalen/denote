# denote.backends.roformer.adapter

Adapter for RoFormer checkpoints via audio-separator, in its own environment.

### Module Attributes

| [`MODEL_DIR_ENVVAR`](#denote.backends.roformer.adapter.MODEL_DIR_ENVVAR)   | audio-separator's own variable for its weights folder (it wins when set).   |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------|

### Functions

| [`checkpoint_license`](#denote.backends.roformer.adapter.checkpoint_license)(model)   | `weights_license` and `commercial_use` of checkpoint `model`.   |
|------------------------------------------------------------------------------|-----------------------------------------------------------------|

### Classes

| [`Adapter`](#denote.backends.roformer.adapter.Adapter)(config)   | RoFormer separation: the default checkpoint gives 'vocals' and 'other'.   |
|--------------------------------------------------------------------|---------------------------------------------------------------------------|

### *class* denote.backends.roformer.adapter.Adapter(config)

Bases: [`IsolatedSeparator`](denote.isolated.html.md#denote.isolated.IsolatedSeparator)

RoFormer separation: the default checkpoint gives ‘vocals’ and ‘other’.

#### separate(audio, , sr=None, \*\*kwargs)

Separate a mix with a RoFormer checkpoint (see `IsolatedSeparator.separate`).

`result.raw` carries the checkpoint’s `weights_license` and
`commercial_use`: “unstated” / “unclear” for one whose card states none.

#### worker_init()

Keyword arguments of the worker’s `load`.

### denote.backends.roformer.adapter.MODEL_DIR_ENVVAR *= 'AUDIO_SEPARATOR_MODEL_DIR'*

audio-separator’s own variable for its weights folder (it wins when set).

### denote.backends.roformer.adapter.checkpoint_license(model)

`weights_license` and `commercial_use` of checkpoint `model`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
