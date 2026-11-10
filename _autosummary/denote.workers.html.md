# denote.workers

Worker files: model code that runs in a backend’s own environment (`denote.isolated`).

Each `<backend>_worker.py` is loaded standalone by `ps.isolated.Worker` with the
environment’s interpreter, so it imports nothing from denote: it defines
`load(**init) -> state` and operations `op(state, **request) -> dict`.

### Modules

| [`demucs_mlx_worker`](denote.workers.demucs_mlx_worker.html.md#module-denote.workers.demucs_mlx_worker)   | demucs-mlx worker: HTDemucs on Apple's MLX, run in the `demucs_mlx` environment.            |
|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`roformer_worker`](denote.workers.roformer_worker.html.md#module-denote.workers.roformer_worker)       | RoFormer worker: audio-separator's RoFormer checkpoints, run in the `roformer` environment. |
