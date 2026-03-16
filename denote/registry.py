"""Backend discovery, registration, and lazy loading."""

import importlib
import pkgutil
import warnings
from typing import Dict, List, Optional, Any

# Module-level registry: backend_name -> {config, adapter (lazy), tasks}
_registry: Dict[str, Dict[str, Any]] = {}

# Tracks whether auto-discovery has run
_discovered = False


def _discover_backends():
    """Scan denote.backends.* for BACKEND_CONFIG dicts. Called once on first access."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    try:
        import denote.backends as backends_pkg
    except ImportError:
        return

    for importer, modname, ispkg in pkgutil.iter_modules(
        backends_pkg.__path__, prefix='denote.backends.'
    ):
        if not ispkg:
            continue
        try:
            config_mod = importlib.import_module(f'{modname}.config')
            config = getattr(config_mod, 'BACKEND_CONFIG', None)
            if config is None:
                continue
            name = config['name']
            _registry[name] = {
                'config': config,
                'adapter': None,  # lazy
                'module_path': modname,
            }
        except Exception as e:
            warnings.warn(
                f"Failed to load backend config from {modname}: {e}",
                stacklevel=2,
            )


def _load_adapter(name: str):
    """Lazily load and instantiate the adapter for a backend."""
    entry = _registry.get(name)
    if entry is None:
        raise KeyError(f"Unknown backend: '{name}'")

    if entry['adapter'] is not None:
        return entry['adapter']

    module_path = entry['module_path']
    config = entry['config']

    try:
        adapter_mod = importlib.import_module(f'{module_path}.adapter')
        adapter_cls = getattr(adapter_mod, 'Adapter')
        entry['adapter'] = adapter_cls(config)
    except ImportError as e:
        pip_install = config.get('pip_install', name)
        raise ImportError(
            f"Backend '{name}' requires: pip install {pip_install}\n"
            f"Original error: {e}"
        ) from e

    return entry['adapter']


def register_backend(
    name: str,
    config: dict,
    adapter: Any = None,
):
    """Register a backend (for third-party plugins).

    Args:
        name: Unique backend identifier.
        config: BACKEND_CONFIG dict with at minimum 'name' and 'tasks'.
        adapter: Optional pre-instantiated adapter. If None, will be loaded
            lazily from the module path if available.
    """
    _registry[name] = {
        'config': config,
        'adapter': adapter,
        'module_path': config.get('module_path', ''),
    }


def get_backend(name: str) -> dict:
    """Get a backend's config and lazily-loaded adapter.

    Returns:
        Dict with 'config' and 'adapter' keys.
    """
    _discover_backends()
    if name not in _registry:
        raise KeyError(
            f"Unknown backend: '{name}'. "
            f"Available: {sorted(_registry.keys())}"
        )
    adapter = _load_adapter(name)
    return {'config': _registry[name]['config'], 'adapter': adapter}


def list_backends(task: Optional[str] = None) -> List[str]:
    """List registered backends, optionally filtered by task.

    Args:
        task: If set, only return backends that support this task
            ('transcribe', 'pitch', 'chords', 'beats', 'separate').

    Returns:
        Sorted list of backend names.
    """
    _discover_backends()
    if task is None:
        return sorted(_registry.keys())
    return sorted(
        name
        for name, entry in _registry.items()
        if task in entry['config'].get('tasks', [])
    )


def get_default_backend(task: str) -> str:
    """Return the default backend name for a given task.

    Looks for backends with 'default_for' containing the task.
    Falls back to the first registered backend for that task.
    """
    _discover_backends()
    # First pass: look for explicit default_for
    for name, entry in _registry.items():
        if task in entry['config'].get('default_for', []):
            return name
    # Second pass: first available backend for the task
    candidates = list_backends(task)
    if not candidates:
        raise ValueError(
            f"No backends registered for task '{task}'. "
            f"Install a backend: pip install denote[basic_pitch]"
        )
    return candidates[0]


def get_config(name: str) -> dict:
    """Get a backend's config without loading its adapter."""
    _discover_backends()
    if name not in _registry:
        raise KeyError(f"Unknown backend: '{name}'")
    return _registry[name]['config']


def clear_registry():
    """Clear all registered backends. Useful for testing."""
    global _discovered
    _registry.clear()
    _discovered = False
