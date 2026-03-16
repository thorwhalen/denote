# Testing — denote

## Test Framework

pytest

## Running Tests

```bash
pytest
```

## Test Structure

- Tests live in `tests/`
- Test files follow `test_{module_name}.py` naming
- Fixtures and shared helpers in `tests/conftest.py`

## Coverage

```bash
pytest --cov
```

## Writing Tests

- Each test function tests one behavior
- Use descriptive test names: `test_{what}_{condition}_{expected}`
- Prefer pytest fixtures over setUp/tearDown
