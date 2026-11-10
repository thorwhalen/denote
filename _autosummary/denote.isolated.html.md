# denote.isolated

Backends that run in their own Python environment, through `ps.isolated`.

Some models cannot share an environment with denote (or with each other):
`demucs-mlx` pins its own MLX, `audio-separator` needs a recent torch. Their
backends declare an `isolated_env` in `BACKEND_CONFIG` (top-level requirements, a
lock file resolved from them, and the Python version) and keep the model code in a
**worker file**
(`denote/workers/<backend>_worker.py`): a plain module, importing nothing from
denote, that `ps.isolated.Worker` runs with the environment’s interpreter. Requests
and replies are JSON lines; audio crosses as `.npy` files in a temporary folder.

Where things live (`DENOTE_DATA_DIR` replaces the root):

```default
~/.local/share/denote/
    envs/<backend>/      the backend's environment (built by uv)
    models/<backend>/    weights a backend keeps itself (when its library has no
                         cache of its own outside a temporary folder)
```

Nothing is built implicitly: a missing environment raises [`EnvNotBuilt`](#denote.isolated.EnvNotBuilt), whose
message is the command that builds it (`python -m denote.isolated build <backend>`),
unless the caller passes `build_env=True`.

An environment is installed from its backend’s **lock** (`env.lock` next to its
config: every package pinned, with platform markers, as `uv pip compile --universal`
resolves the top-level requirements), so it is the same wherever and whenever it is
built, and changing the lock changes the spec’s digest (the next build starts afresh).
`python -m denote.isolated lock <backend>` re-resolves it.

A worker keeps its model in memory (several GB for these models) until it is closed
(`denote.services[<backend>].close()`) or the host process exits.

```pycon
>>> spec = env_spec({"name": "toy", "isolated_env": {"requirements": ["numpy==2.1.0"]}})
>>> spec.name, spec.requirements
('toy', ('numpy==2.1.0',))
>>> build_command("toy")
'python -m denote.isolated build toy'
```

### Module Attributes

| [`DATA_DIR_ENVVAR`](#denote.isolated.DATA_DIR_ENVVAR)   | Environment variable naming denote's data root.                     |
|--------------------------------------------------------------------|---------------------------------------------------------------------|
| [`DFLT_DATA_DIR`](#denote.isolated.DFLT_DATA_DIR)     | The data root when `DENOTE_DATA_DIR` is not set.                    |
| [`DFLT_ENV_PYTHON`](#denote.isolated.DFLT_ENV_PYTHON)   | Python of an isolated environment when its config names none.       |
| [`WORKERS_DIR`](#denote.isolated.WORKERS_DIR)       | The folder holding the worker files.                                |
| [`MIX_FILE`](#denote.isolated.MIX_FILE)          | The file a worker reads its input from, in the request's folder.    |
| [`HOST_UV_VARS`](#denote.isolated.HOST_UV_VARS)      | uv settings holding the host's pins, cleared when resolving a lock. |

### Functions

| [`build_command`](#denote.isolated.build_command)(name)                        | The shell command that builds backend `name`'s environment.                         |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`build_env`](#denote.isolated.build_env)(name, \*\*ensure_kwargs)         | Build backend `name`'s environment (no-op if built from this spec); its Python.     |
| [`compile_lock`](#denote.isolated.compile_lock)(name, \*[, run])              | Re-resolve backend `name`'s lock from its top-level requirements; the lock's path.  |
| [`data_dir`](#denote.isolated.data_dir)(\*parts)                          | `$DENOTE_DATA_DIR/<parts>` (default root `~/.local/share/denote`), not created.     |
| [`env_dir`](#denote.isolated.env_dir)(name)                              | The folder of backend `name`'s environment (it may not exist).                      |
| [`env_python`](#denote.isolated.env_python)(config, \*[, build])            | The interpreter of the backend's environment, building it only if `build`.          |
| [`env_size_hint`](#denote.isolated.env_size_hint)(config)                      | The rough download size a config states for its environment.                        |
| [`env_spec`](#denote.isolated.env_spec)(config)                           | The `ps.isolated.EnvSpec` of a backend config's `isolated_env`.                     |
| [`is_env_ready`](#denote.isolated.is_env_ready)(config)                       | True if the backend's environment exists and was built from its current spec.       |
| [`lock_command`](#denote.isolated.lock_command)(config)                       | The `uv pip compile` command that resolves a config's lock (requirements on stdin). |
| [`lock_requirements`](#denote.isolated.lock_requirements)(text)                    | The requirement lines of a lock (`uv pip compile` output), comments dropped.        |
| [`make_worker`](#denote.isolated.make_worker)(config, \*[, init, build])     | A `ps.isolated.Worker` running the backend's worker file in its environment.        |
| [`separate_in_worker`](#denote.isolated.separate_in_worker)(call, mix, \*\*request) | Run a worker's `separate` on `mix` (`(channels, frames)`), return `(stems, reply)`. |
| [`worker_file`](#denote.isolated.worker_file)(name)                          | The worker file of backend `name`.                                                  |

### Classes

| [`IsolatedSeparator`](#denote.isolated.IsolatedSeparator)(config)   | Host side of a separation backend whose model runs in a worker (see module doc).   |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------|

### Exceptions

| [`EnvNotBuilt`](#denote.isolated.EnvNotBuilt)   | A backend's isolated environment is not built; the message says how to build it.   |
|----------------------------------------------------------------|------------------------------------------------------------------------------------|

### denote.isolated.DATA_DIR_ENVVAR *= 'DENOTE_DATA_DIR'*

Environment variable naming denote’s data root.

### denote.isolated.DFLT_DATA_DIR *= '~/.local/share/denote'*

The data root when `DENOTE_DATA_DIR` is not set.

### denote.isolated.DFLT_ENV_PYTHON *= '3.12'*

Python of an isolated environment when its config names none.

### *exception* denote.isolated.EnvNotBuilt

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

A backend’s isolated environment is not built; the message says how to build it.

### denote.isolated.HOST_UV_VARS *= ('UV_CONSTRAINT', 'UV_OVERRIDE', 'UV_BUILD_CONSTRAINT')*

uv settings holding the host’s pins, cleared when resolving a lock.

### *class* denote.isolated.IsolatedSeparator(config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Host side of a separation backend whose model runs in a worker (see module doc).

Subclasses set [`MODEL_SR`](#denote.isolated.IsolatedSeparator.MODEL_SR) and may override [`worker_init()`](#denote.isolated.IsolatedSeparator.worker_init) (the
worker’s `load` arguments) and [`request()`](#denote.isolated.IsolatedSeparator.request) (one `separate` call’s). The
worker starts on the first call (the model loads once) and is reused; `close()`
stops it. Config needs `param_map` with `model`, `sources` and `build_env`.

#### CHANNELS *= 2*

Channels the model takes (mono is duplicated, extra channels dropped).

#### MODEL_SR *= 44100*

The rate the model runs at; the mix is resampled to it on the host.

#### check(opts)

Refuse bad options before any environment is touched (default: none).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### close()

Stop the worker (the next call starts a fresh one).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### request(opts)

The `separate` request’s options (besides `mix`, `out_dir`, `sr`).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### separate(audio, , sr=None, \*\*kwargs)

Separate a mix into stems in the backend’s own environment.

* **Parameters:**
  * **audio** – File path or numpy array (mono, `(frames, channels)` or
    `(channels, frames)`); mono is duplicated to stereo.
  * **sr** – Sample rate (required if audio is an array).
  * **\*\*kwargs** – Normalized parameters (see the backend’s `param_map`);
    `build_env=True` builds a missing environment instead of raising.
* **Returns:**
  SeparationResult with `(2, frames)` float32 stems at the model’s rate.
* **Raises:**
  [**EnvNotBuilt**](#denote.isolated.EnvNotBuilt) – The environment is not built and `build_env` is false.

#### worker_init()

Keyword arguments of the worker’s `load`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### denote.isolated.MIX_FILE *= 'mix.npy'*

The file a worker reads its input from, in the request’s folder.

### denote.isolated.WORKERS_DIR *= PosixPath('/home/runner/work/denote/denote/denote/workers')*

The folder holding the worker files.

### denote.isolated.build_command(name)

The shell command that builds backend `name`’s environment.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.isolated.build_env(name, \*\*ensure_kwargs)

Build backend `name`’s environment (no-op if built from this spec); its Python.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### denote.isolated.compile_lock(name, , run=None)

Re-resolve backend `name`’s lock from its top-level requirements; the lock’s path.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### denote.isolated.data_dir(\*parts)

`$DENOTE_DATA_DIR/<parts>` (default root `~/.local/share/denote`), not created.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### denote.isolated.env_dir(name)

The folder of backend `name`’s environment (it may not exist).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### denote.isolated.env_python(config, , build=False, \*\*ensure_kwargs)

The interpreter of the backend’s environment, building it only if `build`.

* **Raises:**
  [**EnvNotBuilt**](#denote.isolated.EnvNotBuilt) – The environment is missing or stale and `build` is false.
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### denote.isolated.env_size_hint(config)

The rough download size a config states for its environment.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.isolated.env_spec(config)

The `ps.isolated.EnvSpec` of a backend config’s `isolated_env`.

Its requirements are the lock’s pinned lines when the config names a `lock`
(so the lock’s content is in the digest), else the top-level `requirements`.

### denote.isolated.is_env_ready(config)

True if the backend’s environment exists and was built from its current spec.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### denote.isolated.lock_command(config)

The `uv pip compile` command that resolves a config’s lock (requirements on stdin).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### denote.isolated.lock_requirements(text)

The requirement lines of a lock (`uv pip compile` output), comments dropped.

```pycon
>>> lock_requirements("# header\nnumpy==2.1.0\n    # via x\ntorch==2.15.0 ; sys_platform == 'darwin'\n")
('numpy==2.1.0', "torch==2.15.0 ; sys_platform == 'darwin'")
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)

### denote.isolated.make_worker(config, , init=None, build=False)

A `ps.isolated.Worker` running the backend’s worker file in its environment.

### denote.isolated.separate_in_worker(call, mix, \*\*request)

Run a worker’s `separate` on `mix` (`(channels, frames)`), return `(stems, reply)`.

The mix goes to the worker as `mix.npy` in a temporary folder; the worker writes
one `.npy` per stem there and replies `{"stems": {name: path}, ...}`. The
folder is removed before this returns.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### denote.isolated.worker_file(name)

The worker file of backend `name`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
