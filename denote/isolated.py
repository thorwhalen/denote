"""Backends that run in their own Python environment, through ``ps.isolated``.

Some models cannot share an environment with denote (or with each other):
``demucs-mlx`` pins its own MLX, ``audio-separator`` needs a recent torch. Their
backends declare an ``isolated_env`` in ``BACKEND_CONFIG`` (the pinned requirements and
the Python version) and keep the model code in a **worker file**
(``denote/workers/<backend>_worker.py``): a plain module, importing nothing from
denote, that ``ps.isolated.Worker`` runs with the environment's interpreter. Requests
and replies are JSON lines; audio crosses as ``.npy`` files in a temporary folder.

Where things live (``DENOTE_DATA_DIR`` replaces the root)::

    ~/.local/share/denote/
        envs/<backend>/      the backend's environment (built by uv)
        models/<backend>/    weights a backend keeps itself (when its library has no
                             cache of its own outside a temporary folder)

Nothing is built implicitly: a missing environment raises :class:`EnvNotBuilt`, whose
message is the command that builds it (``python -m denote.isolated build <backend>``),
unless the caller passes ``build_env=True``.

>>> spec = env_spec({"name": "toy", "isolated_env": {"requirements": ["numpy==2.1.0"]}})
>>> spec.name, spec.requirements
('toy', ('numpy==2.1.0',))
>>> build_command("toy")
'python -m denote.isolated build toy'
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

#: Environment variable naming denote's data root.
DATA_DIR_ENVVAR = "DENOTE_DATA_DIR"
#: The data root when ``DENOTE_DATA_DIR`` is not set.
DFLT_DATA_DIR = "~/.local/share/denote"
ENVS_KIND = "envs"
MODELS_KIND = "models"
#: Python of an isolated environment when its config names none.
DFLT_ENV_PYTHON = "3.12"
#: The folder holding the worker files.
WORKERS_DIR = Path(__file__).parent / "workers"
#: The file a worker reads its input from, in the request's folder.
MIX_FILE = "mix.npy"


class EnvNotBuilt(RuntimeError):
    """A backend's isolated environment is not built; the message says how to build it."""


def data_dir(*parts: str) -> Path:
    """``$DENOTE_DATA_DIR/<parts>`` (default root ``~/.local/share/denote``), not created."""
    root = os.environ.get(DATA_DIR_ENVVAR) or DFLT_DATA_DIR
    return Path(root).expanduser().joinpath(*parts)


def env_spec(config: dict):
    """The ``ps.isolated.EnvSpec`` of a backend config's ``isolated_env``."""
    from ps.isolated import EnvSpec

    env = config["isolated_env"]
    return EnvSpec(
        config["name"],
        tuple(env["requirements"]),
        python=env.get("python", DFLT_ENV_PYTHON),
        torch_backend=env.get("torch_backend"),
    )


def env_dir(name: str) -> Path:
    """The folder of backend ``name``'s environment (it may not exist)."""
    return data_dir(ENVS_KIND, name)


def build_command(name: str) -> str:
    """The shell command that builds backend ``name``'s environment."""
    return f"python -m denote.isolated build {name}"


def is_env_ready(config: dict) -> bool:
    """True if the backend's environment exists and was built from its current spec."""
    from ps.isolated import is_env_ready as ready

    return ready(env_spec(config), env_dir(config["name"]))


def env_python(config: dict, *, build: bool = False, **ensure_kwargs) -> Path:
    """The interpreter of the backend's environment, building it only if ``build``.

    Raises:
        EnvNotBuilt: The environment is missing or stale and ``build`` is false.
    """
    from ps.isolated import ensure_env, venv_python

    name, spec = config["name"], env_spec(config)
    if is_env_ready(config):
        return venv_python(env_dir(name))
    if not build:
        raise EnvNotBuilt(
            f"Backend {name!r} runs in its own environment, which is not built "
            f"(or was built from other requirements) at {env_dir(name)}. "
            f"Build it (one-time, about {env_size_hint(config)}) with:\n\n"
            f"    {build_command(name)}\n\n"
            f"or pass build_env=True. ${DATA_DIR_ENVVAR} moves it."
        )
    return ensure_env(spec, env_dir(name), **ensure_kwargs)


def env_size_hint(config: dict) -> str:
    """The rough download size a config states for its environment."""
    return config["isolated_env"].get("size_hint", "a few hundred MB")


def build_env(name: str, **ensure_kwargs) -> Path:
    """Build backend ``name``'s environment (no-op if built from this spec); its Python."""
    from denote.registry import get_config

    return env_python(get_config(name), build=True, **ensure_kwargs)


def worker_file(name: str) -> Path:
    """The worker file of backend ``name``."""
    return WORKERS_DIR / f"{name}_worker.py"


def make_worker(config: dict, *, init: Optional[dict] = None, build: bool = False):
    """A ``ps.isolated.Worker`` running the backend's worker file in its environment."""
    from ps.isolated import Worker

    return Worker(
        env_python(config, build=build),
        worker_file(config["name"]),
        init=init or {},
    )


def separate_in_worker(
    call: Callable[..., dict], mix: np.ndarray, **request
) -> tuple[dict, dict]:
    """Run a worker's ``separate`` on ``mix`` (``(channels, frames)``), return ``(stems, reply)``.

    The mix goes to the worker as ``mix.npy`` in a temporary folder; the worker writes
    one ``.npy`` per stem there and replies ``{"stems": {name: path}, ...}``. The
    folder is removed before this returns.
    """
    folder = tempfile.mkdtemp(prefix="denote-")
    try:
        mix_path = os.path.join(folder, MIX_FILE)
        np.save(mix_path, np.ascontiguousarray(mix, dtype=np.float32))
        reply = call("separate", mix=mix_path, out_dir=folder, **request)
        stems = {name: np.load(path) for name, path in reply["stems"].items()}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return stems, reply


def keep_sources(stems: dict, sources) -> dict:
    """``stems`` restricted to ``sources`` (all when None), in that order."""
    if sources is None:
        return stems
    if unknown := set(sources) - set(stems):
        raise ValueError(
            f"unknown sources {sorted(unknown)}; this model has {list(stems)}"
        )
    return {name: stems[name] for name in sources}


class IsolatedSeparator:
    """Host side of a separation backend whose model runs in a worker (see module doc).

    Subclasses set :attr:`MODEL_SR` and may override :meth:`worker_init` (the
    worker's ``load`` arguments) and :meth:`request` (one ``separate`` call's). The
    worker starts on the first call (the model loads once) and is reused; ``close()``
    stops it. Config needs ``param_map`` with ``model``, ``sources`` and ``build_env``.
    """

    #: The rate the model runs at; the mix is resampled to it on the host.
    MODEL_SR = 44100
    #: Channels the model takes (mono is duplicated, extra channels dropped).
    CHANNELS = 2

    def __init__(self, config: dict):
        from denote.translation import make_kwargs_translator

        self._config = config
        self._translate = make_kwargs_translator(config["param_map"])
        self._worker = None

    def worker_init(self) -> dict:
        """Keyword arguments of the worker's ``load``."""
        return {}

    def request(self, opts: dict) -> dict:
        """The ``separate`` request's options (besides ``mix``, ``out_dir``, ``sr``)."""
        return {"model": opts["model"], "sources": opts["sources"]}

    def check(self, opts: dict) -> None:
        """Refuse bad options before any environment is touched (default: none)."""

    def _worker_for(self, build: bool):
        if self._worker is None:
            self._worker = make_worker(
                self._config, init=self.worker_init(), build=build
            )
        return self._worker

    def separate(self, audio, *, sr=None, **kwargs):
        """Separate a mix into stems in the backend's own environment.

        Args:
            audio: File path or numpy array (mono, ``(frames, channels)`` or
                ``(channels, frames)``); mono is duplicated to stereo.
            sr: Sample rate (required if audio is an array).
            **kwargs: Normalized parameters (see the backend's ``param_map``);
                ``build_env=True`` builds a missing environment instead of raising.

        Returns:
            SeparationResult with ``(2, frames)`` float32 stems at the model's rate.

        Raises:
            EnvNotBuilt: The environment is not built and ``build_env`` is false.
        """
        from denote.base import SeparationResult
        from denote.util import load_audio

        opts = self._translate(**kwargs)
        self.check(opts)
        worker = self._worker_for(opts["build_env"])
        y, _ = load_audio(audio, sr=sr, mono=False, target_sr=self.MODEL_SR)
        x = np.atleast_2d(np.asarray(y, dtype=np.float32))  # (channels, frames)
        x = (
            np.repeat(x, self.CHANNELS, axis=0)
            if x.shape[0] == 1
            else x[: self.CHANNELS]
        )
        stems, reply = separate_in_worker(
            worker.call, x, sr=self.MODEL_SR, **self.request(opts)
        )
        return SeparationResult(
            stems=keep_sources(stems, opts["sources"]),
            sr=reply["sr"],
            raw={k: v for k, v in reply.items() if k != "stems"},
            backend=self._config["name"],
        )

    def close(self) -> None:
        """Stop the worker (the next call starts a fresh one)."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None


def _main(argv=None) -> int:
    """``python -m denote.isolated build <backend> | status``."""
    import argparse

    from denote.registry import get_config, list_backends

    isolated = [n for n in list_backends() if "isolated_env" in get_config(n)]
    parser = argparse.ArgumentParser(prog="python -m denote.isolated")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a backend's environment")
    build.add_argument("backend", choices=isolated)
    sub.add_parser("status", help="where each isolated environment is, and if built")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build_env(args.backend))
    else:
        for name in isolated:
            state = "built" if is_env_ready(get_config(name)) else "not built"
            print(f"{name}: {state} ({env_dir(name)})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
