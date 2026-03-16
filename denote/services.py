"""Service layer providing three tiers of access to backends.

- ServiceCollection: lazy mapping of backend names → ServiceHandle
- ServiceHandle: per-backend namespace with normalized + native access
- SliceMapping: cross-backend view for a specific task
"""

import functools
from collections.abc import Mapping
from typing import Optional, Any

from denote import registry


class ServiceHandle:
    """Per-backend access point with normalized and native methods.

    Attributes:
        name: Backend identifier.
        config: The backend's BACKEND_CONFIG dict.
        adapter: Lazily-loaded adapter instance.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @functools.cached_property
    def config(self) -> dict:
        return registry.get_config(self._name)

    @functools.cached_property
    def adapter(self) -> Any:
        return registry.get_backend(self._name)['adapter']

    @functools.cached_property
    def info(self) -> dict:
        """Lightweight summary of the backend."""
        c = self.config
        return {
            'name': c['name'],
            'display_name': c.get('display_name', c['name']),
            'tasks': c.get('tasks', []),
            'license': c.get('license', 'unknown'),
            'pip_install': c.get('pip_install', ''),
        }

    def __getattr__(self, name: str):
        """Proxy attribute access to the adapter for backend-specific methods."""
        try:
            return getattr(self.adapter, name)
        except AttributeError:
            raise AttributeError(
                f"Backend '{self._name}' has no method '{name}'"
            )

    def __repr__(self):
        tasks = self.config.get('tasks', [])
        return f"<ServiceHandle '{self._name}' tasks={tasks}>"


class ServiceCollection(Mapping):
    """Lazy mapping of backend names → ServiceHandle instances.

    Supports both dict-style and attribute-style access:
        services['basic_pitch']
        services.basic_pitch
    """

    def __init__(self):
        self._handles = {}

    def _ensure_discovered(self):
        registry._discover_backends()

    def __getitem__(self, name: str) -> ServiceHandle:
        self._ensure_discovered()
        if name not in registry._registry:
            raise KeyError(
                f"Unknown backend: '{name}'. "
                f"Available: {sorted(registry._registry.keys())}"
            )
        if name not in self._handles:
            self._handles[name] = ServiceHandle(name)
        return self._handles[name]

    def __getattr__(self, name: str) -> ServiceHandle:
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"No backend named '{name}'. "
                f"Available: {sorted(registry.list_backends())}"
            )

    def __iter__(self):
        self._ensure_discovered()
        return iter(sorted(registry._registry.keys()))

    def __len__(self):
        self._ensure_discovered()
        return len(registry._registry)

    def __contains__(self, name):
        self._ensure_discovered()
        return name in registry._registry

    def __repr__(self):
        backends = list(self)
        return f"<ServiceCollection backends={backends}>"


class SliceMapping(Mapping):
    """Cross-backend view that maps backend names to a specific method.

    Example:
        transcribers = SliceMapping(services, 'transcribe')
        result = transcribers['basic_pitch']("song.wav")
        # equivalent to: services.basic_pitch.transcribe("song.wav")
    """

    def __init__(
        self,
        services: ServiceCollection,
        method_name: str,
        *,
        task_filter: Optional[str] = None,
    ):
        self._services = services
        self._method_name = method_name
        self._task_filter = task_filter

    def _filtered_names(self):
        if self._task_filter:
            return registry.list_backends(self._task_filter)
        return list(self._services)

    def __getitem__(self, name: str):
        handle = self._services[name]
        return getattr(handle, self._method_name)

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        return self[name]

    def __iter__(self):
        return iter(self._filtered_names())

    def __len__(self):
        return len(self._filtered_names())

    def __repr__(self):
        return (
            f"<SliceMapping method='{self._method_name}' "
            f"backends={self._filtered_names()}>"
        )
