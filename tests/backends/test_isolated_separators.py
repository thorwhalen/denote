"""Tests for the separation backends that run in their own environment (thorwhalen/denote#13).

``demucs_mlx`` and ``roformer`` keep their model code in worker files
(``denote/workers/``) that ``ps.isolated.Worker`` runs with another interpreter. Here
the worker files run in this process (or in a real ``Worker`` subprocess of this
interpreter) against fake ``demucs_mlx`` / ``audio_separator`` modules, so neither
library, nor MLX, torch or weights, is needed. The tests at the end separate real
audio only with ``DENOTE_TEST_REAL_ENVS=1`` and the environment built
(``python -m denote.isolated build <backend>``).
"""

import ast
import importlib.util
import itertools
import json
import os
import sys
import types

import numpy as np
import pytest

import denote
from denote import isolated
from denote.base import SeparationResult
from denote.registry import get_config

SR = 44100
FOUR = ("drums", "bass", "other", "vocals")
GAINS = (0.1, 0.2, 0.3, 0.4)  # sum to one: the stems reassemble the input
_names = itertools.count()


def _load_worker_module(path):
    """The worker file at ``path``, imported under a fresh name (as a worker loads it)."""
    spec = importlib.util.spec_from_file_location(f"_test_worker_{next(_names)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InProcessWorker:
    """``ps.isolated.Worker``'s contract, in this process: ``load(**init)`` on the first
    call, then ``op(state, **request)``, requests and results passed through JSON."""

    def __init__(self, python, adapter, *, init=None, **_):
        self.python, self.adapter, self.init = python, adapter, dict(init or {})
        self.module = _load_worker_module(adapter)
        self.state, self.requests, self.closed = None, [], False

    def call(self, op, /, **request):
        request = json.loads(json.dumps(request))  # what can cross the pipe
        if self.state is None:
            self.state = self.module.load(**self.init)
        self.requests.append((op, request))
        result = getattr(self.module, op)(self.state, **request)
        return json.loads(json.dumps({} if result is None else result))

    def close(self):
        self.closed = True


@pytest.fixture
def in_process(monkeypatch):
    """Route ``make_worker`` to in-process workers; record them."""
    made = []

    def make_worker(config, *, init=None, build=False):
        worker = InProcessWorker(
            "env-python", isolated.worker_file(config["name"]), init=init
        )
        worker.build = build
        made.append(worker)
        return worker

    monkeypatch.setattr(isolated, "make_worker", make_worker)
    _fresh_adapters(monkeypatch)
    return made


def _fresh_adapters(monkeypatch, names=("demucs_mlx", "roformer")):
    """New adapters for ``names`` (no worker kept from an earlier test), restored after."""
    from denote import registry

    registry._discover_backends()
    for name in names:
        monkeypatch.setitem(registry._registry[name], "adapter", None)
        monkeypatch.setattr(denote.services, "_handles", dict(denote.services._handles))
        denote.services._handles.pop(name, None)


# --- fake upstream libraries -------------------------------------------------------


#: Model names whose next load fails once (a download error, say).
FAIL_ONCE = set()


def _fake_demucs_mlx(built):
    class Model:
        def __init__(self, name):
            self.sources = list(FOUR) + (
                ["guitar", "piano"] if name.endswith("6s") else []
            )

    class Separator:
        def __init__(self, model="htdemucs", stem=None, **_):
            if model in FAIL_ONCE:
                FAIL_ONCE.discard(model)
                raise ImportError(f"could not download {model}")
            built.append((model, stem))
            self.model, self.stem, self.samplerate = Model(model), stem, SR
            self.shifts, self.overlap, self.segment, self.seed = 1, 0.25, None, None
            self.seen = []  # (shifts, overlap, segment, seed) of each separation

        def update_parameter(self, *, shifts=None, overlap=None, **_):
            # like demucs-mlx: None leaves a value as it was
            self.shifts = self.shifts if shifts is None else shifts
            self.overlap = self.overlap if overlap is None else overlap

        def separate_tensor(self, wav):
            wav = np.asarray(wav)
            assert wav.ndim == 2 and wav.shape[0] == 2
            assert abs(wav.mean()) < 1e-5  # normalized by the worker
            gains = dict(zip(self.model.sources, GAINS + (0.0, 0.0)))
            names = [self.stem] if self.stem else self.model.sources
            self.seen.append((self.shifts, self.overlap, self.segment, self.seed))
            return wav, {n: gains[n] * wav for n in names}

    return types.SimpleNamespace(Separator=Separator)


@pytest.fixture
def fake_demucs(monkeypatch):
    # the worker's load() sets SSL_CERT_FILE only if unset: keep the test process's
    monkeypatch.setenv("SSL_CERT_FILE", "set-by-the-host")
    built = []
    monkeypatch.setitem(sys.modules, "demucs_mlx", _fake_demucs_mlx(built))
    return built


def _fake_audio_separator(loaded):
    import soundfile

    class Separator:
        def __init__(
            self, *, output_dir, model_file_dir, normalization_threshold, **kw
        ):
            self.output_dir, self.model_file_dir = output_dir, model_file_dir
            self.kw = kw
            assert normalization_threshold == 1.0 and kw["use_soundfile"]

        def load_model(self, model_filename):
            if model_filename in FAIL_ONCE:
                FAIL_ONCE.discard(model_filename)
                raise RuntimeError(f"could not download {model_filename}")
            loaded.append(model_filename)
            self.model = model_filename
            self.model_instance = types.SimpleNamespace(primary_stem_name="Vocals")

        def separate(self, path):
            mix, rate = soundfile.read(path, dtype="float32", always_2d=True)
            assert soundfile.info(path).subtype == "FLOAT"
            names = []
            # audio-separator 0.48 writes the secondary stem first
            for stem, gain in (("Other", 0.75), ("Vocals", 0.25)):
                name = f"mix_({stem})_{self.model.split('.')[0]}.wav"
                soundfile.write(
                    os.path.join(self.output_dir, name),
                    gain * mix,
                    rate,
                    subtype="FLOAT",
                )
                names.append(name)  # audio-separator returns names, not paths
            return names

    module = types.ModuleType("audio_separator.separator")
    module.Separator = Separator
    return module


@pytest.fixture
def fake_audio_separator(monkeypatch):
    pytest.importorskip("soundfile")
    loaded = []
    monkeypatch.setitem(
        sys.modules, "audio_separator", types.ModuleType("audio_separator")
    )
    monkeypatch.setitem(
        sys.modules, "audio_separator.separator", _fake_audio_separator(loaded)
    )
    return loaded


def _mix(seconds=1.0, seed=0, offset=0.05):
    rng = np.random.default_rng(seed)
    return rng.normal(offset, 0.1, (2, int(seconds * SR))).astype(np.float32)


# --- demucs_mlx --------------------------------------------------------------------


def test_demucs_mlx_separates_through_its_worker(in_process, fake_demucs, tmp_path):
    mix = _mix()
    result = denote.separate(mix, sr=SR, backend="demucs_mlx", shifts=0)
    assert isinstance(result, SeparationResult) and result.sr == SR
    assert result.backend == "demucs_mlx" and list(result.stems) == list(FOUR)
    for name, gain in zip(FOUR, GAINS):
        assert result.stems[name].dtype == np.float32
    assert np.allclose(sum(result.stems.values()), mix, atol=1e-5)  # mean / n each
    assert result.raw == {"sr": SR, "sources": list(FOUR), "model": "htdemucs"}
    (worker,) = in_process
    op, request = worker.requests[0]
    assert (
        op == "separate" and request["shifts"] == 0 and request["model"] == "htdemucs"
    )
    assert not os.path.exists(request["out_dir"])  # the exchange folder is removed
    assert fake_demucs == [("htdemucs", None)]
    assert os.environ["SSL_CERT_FILE"] == "set-by-the-host"


def test_demucs_mlx_loads_once_and_runs_one_sub_model_for_one_ft_stem(
    in_process, fake_demucs
):
    mix = _mix(0.5)
    denote.separate(mix, sr=SR, backend="demucs_mlx")
    denote.separate(mix, sr=SR, backend="demucs_mlx")
    vocals = denote.separate(
        mix, sr=SR, backend="demucs_mlx", model="htdemucs_ft", sources=("vocals",)
    )
    assert list(vocals.stems) == ["vocals"]
    assert fake_demucs == [("htdemucs", None), ("htdemucs_ft", "vocals")]
    assert len(in_process) == 1  # one worker, kept


def test_demucs_mlx_mono_and_frames_first_input_and_resampling(in_process, fake_demucs):
    mono = _mix(0.5)[0]  # 22050 samples: one second at 22.05 kHz
    result = denote.separate(mono, sr=22050, backend="demucs_mlx")
    assert result.stems["vocals"].shape == (2, SR)
    frames_first = _mix(0.5).T
    assert denote.separate(frames_first, sr=SR, backend="demucs_mlx").stems[
        "drums"
    ].shape == (2, SR // 2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"model": "mdx_extra"}, "model must be one of"),
        ({"sources": ("guitar",)}, "unknown sources"),  # only htdemucs_6s has it
        ({"overlap": 1.0}, "overlap"),
    ],
)
def test_demucs_mlx_refuses_bad_options_before_any_environment(
    in_process, kwargs, match
):
    with pytest.raises(ValueError, match=match):
        denote.separate(_mix(0.1), sr=SR, backend="demucs_mlx", **kwargs)
    assert in_process == []


def test_demucs_mlx_six_stems(in_process, fake_demucs):
    result = denote.separate(
        _mix(0.2),
        sr=SR,
        backend="demucs_mlx",
        model="htdemucs_6s",
        sources=("piano", "vocals"),
    )
    assert list(result.stems) == ["piano", "vocals"]


# --- roformer ----------------------------------------------------------------------


def test_roformer_separates_through_its_worker(
    in_process, fake_audio_separator, tmp_path, monkeypatch
):
    monkeypatch.setenv(isolated.DATA_DIR_ENVVAR, str(tmp_path))
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    mix = _mix(offset=0.0)
    result = denote.separate(mix, sr=SR, backend="roformer")
    assert list(result.stems) == ["vocals", "other"]  # lower case
    assert np.allclose(result.stems["vocals"], 0.25 * mix, atol=1e-6)
    assert np.allclose(sum(result.stems.values()), mix, atol=1e-6)
    assert result.raw["model"] == "vocals_mel_band_roformer.ckpt"
    assert (result.raw["weights_license"], result.raw["commercial_use"]) == (
        "MIT",
        "yes",
    )
    (worker,) = in_process
    assert worker.init == {"model_file_dir": str(tmp_path / "models" / "roformer")}
    assert os.listdir(worker.state["outputs"]) == []  # wavs removed after each call


def test_roformer_honours_audio_separators_own_model_dir(
    in_process, fake_audio_separator, tmp_path, monkeypatch
):
    monkeypatch.setenv(isolated.DATA_DIR_ENVVAR, str(tmp_path / "data"))
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "theirs"))
    denote.separate(_mix(0.1, offset=0.0), sr=SR, backend="roformer")
    assert in_process[0].init == {"model_file_dir": str(tmp_path / "theirs")}


def test_roformer_other_checkpoints_are_unclear_and_load_on_demand(
    in_process, fake_audio_separator
):
    mix = _mix(0.2, offset=0.0)
    denote.separate(mix, sr=SR, backend="roformer")
    result = denote.separate(
        mix,
        sr=SR,
        backend="roformer",
        model="bs_roformer_vocals_gabox.ckpt",
        sources=["vocals"],
    )
    assert list(result.stems) == ["vocals"]
    assert (result.raw["weights_license"], result.raw["commercial_use"]) == (
        "unstated",
        "unclear",
    )
    assert fake_audio_separator == [
        "vocals_mel_band_roformer.ckpt",
        "bs_roformer_vocals_gabox.ckpt",
    ]


def test_roformer_unknown_source_is_an_error(in_process, fake_audio_separator):
    with pytest.raises(ValueError, match="unknown sources"):
        denote.separate(_mix(0.1), sr=SR, backend="roformer", sources=["drums"])


def test_roformer_stem_name_parsing():
    worker = _load_worker_module(isolated.worker_file("roformer"))
    assert worker.stem_name("/x/mix_(Instrumental)_model_bs.wav") == "instrumental"
    with pytest.raises(ValueError):
        worker.stem_name("mix.wav")


# --- environments ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["demucs_mlx", "roformer"])
def test_a_missing_environment_raises_with_the_build_command(
    name, tmp_path, monkeypatch
):
    monkeypatch.setenv(isolated.DATA_DIR_ENVVAR, str(tmp_path))
    _fresh_adapters(monkeypatch, [name])
    with pytest.raises(isolated.EnvNotBuilt) as info:
        denote.separate(_mix(0.1), sr=SR, backend=name)
    assert f"python -m denote.isolated build {name}" in str(info.value)
    assert str(tmp_path / "envs" / name) in str(info.value)


def test_build_env_builds_with_the_pinned_spec_under_the_data_dir(
    tmp_path, monkeypatch
):
    from ps import isolated as ps_isolated

    monkeypatch.setenv(isolated.DATA_DIR_ENVVAR, str(tmp_path))
    built = []

    def ensure_env(spec, path, **kw):
        built.append((spec, path))
        return path / "bin" / "python"

    monkeypatch.setattr(ps_isolated, "ensure_env", ensure_env)
    python = isolated.build_env("demucs_mlx")
    ((spec, path),) = built
    assert path == tmp_path / "envs" / "demucs_mlx" and python.parent.parent == path
    assert spec.python == "3.12"
    # installed from the lock: every package pinned, the lock's content in the digest
    assert "demucs-mlx==1.6.1" in spec.requirements
    assert all("==" in line for line in spec.requirements)
    pinned = {line.split("==")[0] for line in spec.requirements}
    assert {"mlx", "certifi", "numpy"} <= pinned
    roformer = isolated.env_spec(get_config("roformer")).requirements
    assert "audio-separator==0.48.0" in roformer  # the [cpu] extra resolved below
    assert {"torch", "onnx-weekly", "onnxruntime", "audioread", "certifi"} <= {
        line.split("==")[0] for line in roformer
    }


@pytest.mark.parametrize("name", ["demucs_mlx", "roformer"])
def test_each_lock_resolves_the_top_level_requirements(name):
    env = get_config(name)["isolated_env"]
    locked = {
        line.split(" ;")[0] for line in isolated.env_spec(get_config(name)).requirements
    }
    for requirement in env["requirements"]:
        package = requirement.split("[")[0].split("=")[0].split(">")[0]
        assert any(line.startswith(package + "==") for line in locked), requirement
    assert f"python -m denote.isolated lock {name}" in open(env["lock"]).read()


def test_a_changed_lock_changes_the_spec_digest(tmp_path):
    lock = tmp_path / "env.lock"
    config = {
        "name": "toy",
        "isolated_env": {"requirements": ["numpy"], "lock": str(lock)},
    }
    lock.write_text("numpy==2.1.0\n    # via toy\n")
    before = isolated.env_spec(config)
    lock.write_text("numpy==2.1.1\n")
    after = isolated.env_spec(config)
    assert (
        before.requirements == ("numpy==2.1.0",) and before.digest() != after.digest()
    )


def test_compile_lock_runs_uv_pip_compile_universal_with_the_requirements(monkeypatch):
    monkeypatch.setenv("UV_CONSTRAINT", "/host/pins.txt")
    calls = []

    def run(command, **kw):
        calls.append((command, kw))

    path = isolated.compile_lock("roformer", run=run)
    ((command, kw),) = calls
    assert command[:5] == ["uv", "pip", "compile", "-", "--universal"]
    assert command[command.index("--output-file") + 1] == str(path)
    assert path.name == "env.lock" and path.parent.name == "roformer"
    assert (
        kw["input"].splitlines()
        == get_config("roformer")["isolated_env"]["requirements"]
    )
    assert kw["env"]["UV_CONSTRAINT"] == ""  # the host's pins stay out


def test_status_command_lists_the_isolated_backends(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(isolated.DATA_DIR_ENVVAR, str(tmp_path))
    assert isolated._main(["status"]) == 0
    out = capsys.readouterr().out
    assert "demucs_mlx: not built" in out and "roformer: not built" in out


@pytest.mark.parametrize("name", ["demucs_mlx", "roformer"])
def test_worker_files_import_nothing_from_denote(name):
    """They run in environments that need not have denote."""
    tree = ast.parse(isolated.worker_file(name).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "denote" not in imported


def test_a_real_worker_process_runs_the_demucs_worker_file(tmp_path):
    """The JSON-lines exchange, end to end, with a fake demucs_mlx on the worker's path."""
    from ps.isolated import Worker

    package = tmp_path / "fake" / "demucs_mlx"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        "import numpy as np\n"
        "class _Model:\n"
        "    sources = ['drums', 'bass', 'other', 'vocals']\n"
        "class Separator:\n"
        "    samplerate = 44100\n"
        "    def __init__(self, model, stem=None): self.model = _Model()\n"
        "    def update_parameter(self, **kw): pass\n"
        "    def separate_tensor(self, wav):\n"
        "        return wav, {n: 0.25 * wav for n in self.model.sources}\n"
    )
    mix = _mix(0.2, offset=0.0)
    with Worker(
        sys.executable,
        isolated.worker_file("demucs_mlx"),
        env={"PYTHONPATH": str(tmp_path / "fake")},
    ) as worker:
        stems, reply = isolated.separate_in_worker(
            worker.call, mix, sr=SR, model="htdemucs", sources=["vocals", "bass"]
        )
    assert list(stems) == ["vocals", "bass"] and reply["model"] == "htdemucs"
    assert np.allclose(stems["vocals"], 0.25 * mix, atol=1e-6)


# --- real environments (opt-in) ----------------------------------------------------

REAL = os.environ.get("DENOTE_TEST_REAL_ENVS") == "1"


def _sdr(reference, estimate):
    return 10 * np.log10(np.sum(reference**2) / np.sum((reference - estimate) ** 2))


@pytest.mark.parametrize("name", ["demucs_mlx", "roformer"])
def test_real_environment_separates_a_synthetic_mix(name):
    """A sine melody over a bass tone: both models file the melody under 'other'
    (it is no voice), so what is checked is that the stems rebuild the mix, and
    that HTDemucs isolates the bass."""
    if not REAL:
        pytest.skip("set DENOTE_TEST_REAL_ENVS=1 to run models in their environments")
    if not isolated.is_env_ready(get_config(name)):
        pytest.skip(f"environment not built: {isolated.build_command(name)}")
    t = np.arange(4 * SR) / SR
    melody = 0.2 * np.sin(2 * np.pi * (440 * t + 3 * np.sin(2 * np.pi * 5 * t)))
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    mix = np.stack([melody + bass] * 2).astype(np.float32)
    try:
        result = denote.separate(mix, sr=SR, backend=name)
    finally:
        denote.services[name].close()
    assert result.sr == SR and all(s.shape == mix.shape for s in result.stems.values())
    assert _sdr(mix, sum(result.stems.values())) > 20
    if name == "demucs_mlx":
        assert list(result.stems) == ["drums", "bass", "other", "vocals"]
        assert _sdr(bass, result.stems["bass"].mean(axis=0)) > 10
    else:
        assert list(result.stems) == ["vocals", "other"]  # target stem first


# --- worker behaviour the facade tests do not reach ---------------------------------


def _worker(name):
    return _load_worker_module(isolated.worker_file(name))


def _request_folder(tmp_path, mix):
    np.save(tmp_path / "mix.npy", mix)
    return {"mix": str(tmp_path / "mix.npy"), "out_dir": str(tmp_path)}


def test_demucs_worker_retries_a_load_that_failed(fake_demucs, tmp_path):
    worker = _worker("demucs_mlx")
    state = worker.load()
    request = _request_folder(tmp_path, _mix(0.1))
    FAIL_ONCE.add("htdemucs")
    with pytest.raises(ImportError, match="could not download"):
        worker.separate(state, sr=SR, **request)
    assert state == {"key": None, "separator": None}  # nothing stale left
    reply = worker.separate(state, sr=SR, **request)
    assert reply["model"] == "htdemucs" and fake_demucs == [("htdemucs", None)]


def test_roformer_worker_retries_a_load_that_failed(fake_audio_separator, tmp_path):
    worker = _worker("roformer")
    state = worker.load(model_file_dir=str(tmp_path / "models"))
    request = _request_folder(tmp_path, _mix(0.1, offset=0.0))
    FAIL_ONCE.add("vocals_mel_band_roformer.ckpt")
    with pytest.raises(RuntimeError, match="could not download"):
        worker.separate(state, sr=SR, model="vocals_mel_band_roformer.ckpt", **request)
    assert state["model"] is None and state["separator"] is None
    reply = worker.separate(
        state, sr=SR, model="vocals_mel_band_roformer.ckpt", **request
    )
    assert list(reply["stems"]) == ["vocals", "other"]


def test_demucs_worker_passes_each_requests_options_and_none_resets(
    fake_demucs, tmp_path
):
    worker = _worker("demucs_mlx")
    state = worker.load()
    request = _request_folder(tmp_path, _mix(0.1))
    worker.separate(state, sr=SR, shifts=2, overlap=0.5, segment=7.0, seed=3, **request)
    worker.separate(state, sr=SR, **request)  # defaults: no segment, no seed
    assert state["separator"].seen == [(2, 0.5, 7.0, 3), (1, 0.25, None, None)]


def test_demucs_worker_writes_only_the_sources_asked_for_in_that_order(
    fake_demucs, tmp_path
):
    worker = _worker("demucs_mlx")
    request = _request_folder(tmp_path, _mix(0.1))
    reply = worker.separate(
        worker.load(), sr=SR, sources=["vocals", "drums"], **request
    )
    assert list(reply["stems"]) == ["vocals", "drums"]
    assert sorted(os.listdir(tmp_path)) == ["drums.npy", "mix.npy", "vocals.npy"]


@pytest.mark.parametrize("name", ["demucs_mlx", "roformer"])
def test_workers_refuse_a_mix_at_another_rate(
    name, fake_demucs, fake_audio_separator, tmp_path
):
    worker = _worker(name)
    init = {"model_file_dir": str(tmp_path)} if name == "roformer" else {}
    request = _request_folder(tmp_path, _mix(0.1))
    with pytest.raises(ValueError, match="44100"):
        worker.separate(worker.load(**init), sr=22050, model="m", **request)


def test_demucs_worker_points_ssl_at_certifi_only_when_unset(monkeypatch):
    certifi = types.SimpleNamespace(where=lambda: "/certifi/cacert.pem")
    monkeypatch.setitem(sys.modules, "certifi", certifi)
    monkeypatch.setenv("SSL_CERT_FILE", "placeholder")
    monkeypatch.delenv("SSL_CERT_FILE")  # restored by monkeypatch afterwards
    _worker("demucs_mlx").load()
    assert os.environ["SSL_CERT_FILE"] == "/certifi/cacert.pem"
    monkeypatch.setenv("SSL_CERT_FILE", "/the/hosts.pem")
    _worker("demucs_mlx").load()
    assert os.environ["SSL_CERT_FILE"] == "/the/hosts.pem"


def test_close_stops_the_worker_and_the_next_call_starts_a_fresh_one(
    in_process, fake_demucs
):
    mix = _mix(0.1)
    denote.separate(mix, sr=SR, backend="demucs_mlx")
    denote.services["demucs_mlx"].close()
    denote.separate(mix, sr=SR, backend="demucs_mlx")
    first, second = in_process
    assert first.closed and not second.closed and len(second.requests) == 1
