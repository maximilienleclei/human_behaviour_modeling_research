"""Project directory paths.

Defines global path constants for data and results directories.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Directory paths
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)
