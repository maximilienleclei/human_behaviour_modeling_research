# Beartype Custom Validators

Custom beartype validator functions for enhanced type checking beyond standard Python types.

## What it's for

Provides reusable type validators for common constraints (non-empty strings/lists, value comparisons, set membership) that integrate with beartype's runtime type checking system. Used throughout the codebase via type annotations.

## What it contains

### Validators
- `not_empty()` - Validates strings or lists are non-empty (length > 0)
- `equal()` - Validates value equals specific element
- `one_of()` - Validates value is in specified set of elements
- `ge()` - Validates numeric value is greater than or equal to threshold
- `gt()` - Validates numeric value is greater than threshold
- `le()` - Validates numeric value is less than or equal to threshold
- `lt()` - Validates numeric value is less than threshold

## Key Details

All validators return `BeartypeValidator` objects that can be used in type annotations with `Annotated`. Example usage:
```python
from typing import Annotated as An
from utils.beartype import ge, one_of

def foo(x: An[int, ge(0)], role: An[str, one_of("input", "hidden", "output")]) -> None:
    ...
```

The validators are used extensively in common/dynamic_net/evolution.py for node roles and UIDs. Beartype enforcement is enabled via `beartype_this_package()` calls in package __init__.py files. These validators catch invalid values at runtime, providing better error messages than assertions.
