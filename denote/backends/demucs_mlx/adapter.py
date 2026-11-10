"""Adapter for demucs-mlx: HTDemucs on MLX, in its own environment (``denote.isolated``)."""

from denote.isolated import IsolatedSeparator


class Adapter(IsolatedSeparator):
    """HTDemucs (v4) on MLX: drums, bass, other, vocals (+ guitar, piano with htdemucs_6s)."""

    def check(self, opts):
        models = self._config["models"]
        if opts["model"] not in models:  # before any download
            raise ValueError(
                f"model must be one of {sorted(models)}, got {opts['model']!r}"
            )
        stems = models[opts["model"]]
        if opts["sources"] is not None:
            if unknown := set(opts["sources"]) - set(stems):
                raise ValueError(
                    f"unknown sources {sorted(unknown)}; {opts['model']} has {list(stems)}"
                )
        if not 0 <= opts["overlap"] < 1:
            raise ValueError(f"overlap must be in [0, 1), got {opts['overlap']}")

    def request(self, opts):
        keys = ("model", "sources", "shifts", "overlap", "seed")
        return {k: opts[k] for k in keys}
