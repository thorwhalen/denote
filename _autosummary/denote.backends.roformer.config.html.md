# denote.backends.roformer.config

Configuration for the RoFormer separation backend (audio-separator).

### Module Attributes

| [`ENV_LOCK`](#denote.backends.roformer.config.ENV_LOCK)            | Every package of the environment, pinned (`python -m denote.isolated lock roformer`).   |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`DFLT_CHECKPOINT`](#denote.backends.roformer.config.DFLT_CHECKPOINT)     | Kimberley Jensen's Mel-Band RoFormer vocals model.                                      |
| [`CHECKPOINTS`](#denote.backends.roformer.config.CHECKPOINTS)         | Weights licence per checkpoint, for checkpoints whose card states one.                  |
| [`UNLISTED_CHECKPOINT`](#denote.backends.roformer.config.UNLISTED_CHECKPOINT) | What is known of a community checkpoint whose card states no licence.                   |

### denote.backends.roformer.config.CHECKPOINTS *= {'vocals_mel_band_roformer.ckpt': {'commercial_use': 'yes', 'sha256': '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e', 'source': 'https://huggingface.co/KimberleyJSN/melbandroformer', 'weights_license': 'MIT'}}*

Weights licence per checkpoint, for checkpoints whose card states one. Any other
checkpoint is reported with `UNLISTED_CHECKPOINT`.

### denote.backends.roformer.config.DFLT_CHECKPOINT *= 'vocals_mel_band_roformer.ckpt'*

Kimberley Jensen’s Mel-Band RoFormer
vocals model. audio-separator’s `vocals_mel_band_roformer.ckpt` is the file
published at [https://huggingface.co/KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer)
(`MelBandRoformer.ckpt`, same SHA-256 below), whose model card states
`license: mit`. (In UVR’s naming, “Kim Vocal 2” is a different, MDX-Net model.)

* **Type:**
  The checkpoint used when none is named

### denote.backends.roformer.config.ENV_LOCK *= PosixPath('/home/runner/work/denote/denote/denote/backends/roformer/env.lock')*

Every package of the environment, pinned (`python -m denote.isolated lock roformer`).

### denote.backends.roformer.config.UNLISTED_CHECKPOINT *= {'commercial_use': 'unclear', 'weights_license': 'unstated'}*

What is known of a community checkpoint whose card states no licence.
