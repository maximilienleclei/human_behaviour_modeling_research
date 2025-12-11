# Data Directory

Human behavioral data collection, replay, and analysis tools.

## Overview

This directory contains tools for the complete data lifecycle: collecting human gameplay data, replaying episodes deterministically, and analyzing session statistics.

## Files

### Data Collection
- **collect.py** - Interactive data collection tool
  - GUI-based human gameplay with keyboard controls
  - Automatic episode recording (observations, actions, rewards, timestamps, seeds)
  - Paused episode-by-episode mode with seed tracking
  - Supports: CartPole, MountainCar, Acrobot, LunarLander
  - Saves to JSON format compatible with platform/data/loaders.py

### Data Replay
- **replay.py** - Deterministic episode replay (note: not documented per plan, <30 lines)
  - Seed-based replay for verification
  - Useful for debugging and visualization

### Data Analysis
- **data_analysis.py** - Session statistics and visualization
  - Session segmentation (30-minute threshold)
  - Episode return and length plotting
  - Multi-file analysis with session boundary visualization
  - Generates publication-ready plots

## Data Format

JSON files contain episode lists with structure:
```json
{
  "timestamp": "ISO-8601 timestamp",
  "seed": integer,
  "steps": [{"observation": [...], "action": int, "reward": float}, ...],
  "return": float
}
```

## Workflow

1. **Collection**: Human plays via collect.py → JSON files saved
2. **Analysis**: data_analysis.py processes files → statistics and plots
3. **Training**: platform/data/loaders.py reads JSON → PyTorch tensors
4. **Verification**: replay.py can recreate episodes from seeds (optional)

## Usage

Data collection is typically done once per subject/environment pair. Analysis helps understand training data characteristics before modeling.
