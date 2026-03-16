"""Parameter translation between normalized and native backend interfaces."""

import warnings
from typing import Dict, Callable, Any, Optional


def make_kwargs_translator(
    param_map: Dict[str, Optional[dict]],
    *,
    on_unsupported: str = 'warn',
) -> Callable:
    """Create a function that translates normalized kwargs to native kwargs.

    Args:
        param_map: Mapping of normalized_name -> native config dict.
            Each config dict can have:
            - 'native_name': str — the backend's parameter name
            - 'coerce': callable — transform the value
            - 'default': Any — default value if not provided
            - None — parameter is not supported by this backend
        on_unsupported: What to do with params not in param_map.
            'warn' (default), 'raise', or 'ignore'.

    Returns:
        A function that translates kwargs dicts.
    """
    supported_names = {k for k, v in param_map.items() if v is not None}

    def translate(**kwargs) -> dict:
        native_kwargs = {}

        for name, value in kwargs.items():
            if name not in param_map:
                if on_unsupported == 'raise':
                    raise ValueError(
                        f"Unsupported parameter: '{name}'. "
                        f"Supported: {sorted(supported_names)}"
                    )
                elif on_unsupported == 'warn':
                    warnings.warn(
                        f"Parameter '{name}' is not supported by this backend "
                        f"and will be ignored.",
                        stacklevel=3,
                    )
                continue

            config = param_map[name]
            if config is None:
                # Explicitly unsupported
                if on_unsupported == 'warn':
                    warnings.warn(
                        f"Parameter '{name}' is not supported by this backend.",
                        stacklevel=3,
                    )
                continue

            native_name = config.get('native_name', name)
            coerce = config.get('coerce')
            if coerce is not None:
                value = coerce(value)
            native_kwargs[native_name] = value

        # Apply defaults for params not provided
        for name, config in param_map.items():
            if config is None:
                continue
            native_name = config.get('native_name', name)
            if native_name not in native_kwargs and 'default' in config:
                native_kwargs[native_name] = config['default']

        return native_kwargs

    return translate


def validate_param(name: str, value: Any, config: dict) -> Any:
    """Validate a single parameter against its config constraints.

    Checks 'min', 'max', and 'choices' constraints.
    """
    if 'min' in config and value < config['min']:
        raise ValueError(
            f"Parameter '{name}' value {value} is below minimum {config['min']}"
        )
    if 'max' in config and value > config['max']:
        raise ValueError(
            f"Parameter '{name}' value {value} is above maximum {config['max']}"
        )
    if 'choices' in config and value not in config['choices']:
        raise ValueError(
            f"Parameter '{name}' value {value!r} not in {config['choices']}"
        )
    return value
