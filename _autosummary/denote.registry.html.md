# denote.registry

Backend discovery, registration, and lazy loading.

### Functions

| [`clear_registry`](#denote.registry.clear_registry)()                          | Clear all registered backends.                         |
|--------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`get_backend`](#denote.registry.get_backend)(name)                         | Get a backend's config and lazily-loaded adapter.      |
| [`get_config`](#denote.registry.get_config)(name)                          | Get a backend's config without loading its adapter.    |
| [`get_default_backend`](#denote.registry.get_default_backend)(task)                 | Return the default backend name for a given task.      |
| [`list_backends`](#denote.registry.list_backends)([task])                     | List registered backends, optionally filtered by task. |
| [`register_backend`](#denote.registry.register_backend)(name, config[, adapter]) | Register a backend (for third-party plugins).          |

### denote.registry.clear_registry()

Clear all registered backends. Useful for testing.

### denote.registry.get_backend(name)

Get a backend’s config and lazily-loaded adapter.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict with ‘config’ and ‘adapter’ keys.

### denote.registry.get_config(name)

Get a backend’s config without loading its adapter.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### denote.registry.get_default_backend(task)

Return the default backend name for a given task.

Looks for backends with ‘default_for’ containing the task.
Falls back to the first registered backend for that task.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### denote.registry.list_backends(task=None)

List registered backends, optionally filtered by task.

* **Parameters:**
  **task** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – If set, only return backends that support this task
  (‘transcribe’, ‘pitch’, ‘chords’, ‘beats’, ‘separate’).
* **Return type:**
  [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  Sorted list of backend names.

### denote.registry.register_backend(name, config, adapter=None)

Register a backend (for third-party plugins).

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Unique backend identifier.
  * **config** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – BACKEND_CONFIG dict with at minimum ‘name’ and ‘tasks’.
  * **adapter** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Optional pre-instantiated adapter. If None, will be loaded
    lazily from the module path if available.
