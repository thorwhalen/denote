"""Adapter for RoFormer checkpoints via audio-separator, in its own environment."""

import os

from denote.isolated import MODELS_KIND, IsolatedSeparator, data_dir

from .config import CHECKPOINTS, UNLISTED_CHECKPOINT

#: audio-separator's own variable for its weights folder (it wins when set).
MODEL_DIR_ENVVAR = "AUDIO_SEPARATOR_MODEL_DIR"


def checkpoint_license(model: str) -> dict:
    """``weights_license`` and ``commercial_use`` of checkpoint ``model``."""
    known = CHECKPOINTS.get(model, UNLISTED_CHECKPOINT)
    return {k: known[k] for k in ("weights_license", "commercial_use")}


class Adapter(IsolatedSeparator):
    """RoFormer separation: the default checkpoint gives 'vocals' and 'other'."""

    def worker_init(self):
        folder = os.environ.get(MODEL_DIR_ENVVAR) or data_dir(MODELS_KIND, "roformer")
        return {"model_file_dir": str(folder)}

    def separate(self, audio, *, sr=None, **kwargs):
        """Separate a mix with a RoFormer checkpoint (see ``IsolatedSeparator.separate``).

        ``result.raw`` carries the checkpoint's ``weights_license`` and
        ``commercial_use``: "unstated" / "unclear" for one whose card states none.
        """
        result = super().separate(audio, sr=sr, **kwargs)
        result.raw.update(checkpoint_license(result.raw["model"]))
        return result
