"""Worker files: model code that runs in a backend's own environment (``denote.isolated``).

Each ``<backend>_worker.py`` is loaded standalone by ``ps.isolated.Worker`` with the
environment's interpreter, so it imports nothing from denote: it defines
``load(**init) -> state`` and operations ``op(state, **request) -> dict``.
"""
