# SimExp Control Tasks

Human behavioral data collected from SimExp control task experiments.

## Purpose

Contains human gameplay episodes from four control tasks (CartPole, MountainCar, Acrobot, LunarLander) and provides loaders, configurations, and preprocessing for accessing this data with continual learning features.

## Structure

### source/
Raw data files and collection utilities:
- JSON data files (sub01_data_*.json, sub02_data_*.json)
- Data collection scripts (collect.py)
- Replay utilities (replay.py)
- Analysis tools (data_analysis.py)

### Root level
- `environments.py` - Environment configurations mapping to data files
- `loaders.py` - Data loading with CL features and train/test splitting
- `preprocessing.py` - Temporal segmentation, normalization, episode datasets

## Data Format

JSON files contain episode lists with:
- observations: Environment state at each timestep
- actions: Human actions taken
- rewards: Rewards received
- timestamp: ISO format timestamp for temporal analysis
- seed_used: Environment random seed

## Usage

Import `load_human_data()` to load data with optional continual learning features (session/run IDs). The loader handles temporal segmentation, normalization, and per-session train/test splitting automatically. Use `EpisodeDataset` for recurrent model training that preserves episode boundaries.
