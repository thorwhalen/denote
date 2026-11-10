# denote.backends.demucs_mlx.adapter

Adapter for demucs-mlx: HTDemucs on MLX, in its own environment (`denote.isolated`).

### Classes

| [`Adapter`](#denote.backends.demucs_mlx.adapter.Adapter)(config)   | HTDemucs (v4) on MLX: drums, bass, other, vocals (+ guitar, piano with htdemucs_6s).   |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------|

### *class* denote.backends.demucs_mlx.adapter.Adapter(config)

Bases: [`IsolatedSeparator`](denote.isolated.html.md#denote.isolated.IsolatedSeparator)

HTDemucs (v4) on MLX: drums, bass, other, vocals (+ guitar, piano with htdemucs_6s).

#### check(opts)

Refuse bad options before any environment is touched (default: none).

#### request(opts)

The `separate` request’s options (besides `mix`, `out_dir`, `sr`).
