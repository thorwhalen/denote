# denote.translation

Parameter translation between normalized and native backend interfaces.

### Functions

| [`make_kwargs_translator`](#denote.translation.make_kwargs_translator)(param_map, \*[, ...])   | Create a function that translates normalized kwargs to native kwargs.   |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`validate_param`](#denote.translation.validate_param)(name, value, config)            | Validate a single parameter against its config constraints.             |

### denote.translation.make_kwargs_translator(param_map, , on_unsupported='warn')

Create a function that translates normalized kwargs to native kwargs.

* **Parameters:**
  * **param_map** ([`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]) – 

    Mapping of normalized_name -> native config dict.
    Each config dict can have:
    - ’native_name’: str — the backend’s parameter name
    - ’coerce’: callable — transform the value
    - ’default’: Any — default value if not provided
    - None — parameter is not supported by this backend
  * **on_unsupported** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – What to do with params not in param_map.
    ‘warn’ (default), ‘raise’, or ‘ignore’.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
* **Returns:**
  A function that translates kwargs dicts.

### denote.translation.validate_param(name, value, config)

Validate a single parameter against its config constraints.

Checks ‘min’, ‘max’, and ‘choices’ constraints.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
