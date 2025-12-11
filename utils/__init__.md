# Utils Directory

Utility functions and helper modules used across the codebase.

## Overview

This directory contains supporting utilities that don't fit into the main platform, experiments, or data categories. Currently focused on type safety infrastructure.

## Files

### Type Safety
- **beartype.py** - Custom beartype validators for runtime type checking
  - Provides validators: `not_empty()`, `equal()`, `one_of()`, `ge()`, `gt()`, `le()`, `lt()`
  - Used extensively with type annotations throughout the codebase
  - Enables rich runtime validation beyond standard Python types

## Usage

The beartype validators are used via type annotations:
```python
from typing import Annotated as An
from utils.beartype import ge, one_of

def process(value: An[int, ge(0)], mode: An[str, one_of("train", "test")]) -> None:
    # value guaranteed to be >= 0
    # mode guaranteed to be "train" or "test"
    pass
```

Beartype enforcement is enabled at the package level via `beartype_this_package()` calls in various `__init__.py` files.

## Future Extensions

This directory can house additional cross-cutting utilities:
- Logging helpers
- Configuration utilities
- Path management
- Common data transformations
