# Platform Data Directory

Data loading and preprocessing functions for training behavior models.

## Overview

This directory handles all data I/O and transformation: loading from HuggingFace and local JSON files, computing temporal structure (sessions/runs), normalizing features, and creating PyTorch datasets.

## Files

### Data Loading
- **loaders.py** - Load data from various sources
  - `load_cartpole_data()` - HuggingFace CartPole-v1 dataset
  - `load_lunarlander_data()` - HuggingFace LunarLander-v2 dataset
  - `load_human_data()` - Local JSON files with human behavioral data
  - Handles train/test splitting and optional continual learning features

### Data Preprocessing
- **preprocessing.py** - Transform raw data for training
  - `compute_session_run_ids()` - Temporal segmentation (30-minute threshold)
  - `normalize_session_run_features()` - Normalize session/run IDs to [0,1]
  - `EpisodeDataset` - PyTorch Dataset preserving episode structure
  - `episode_collate_fn()` - Batch episodes with padding for recurrent models

## Data Flow

1. **Loading**: Raw data loaded from HuggingFace or JSON files
2. **Temporal Features**: Sessions and runs computed from timestamps (if human data)
3. **Normalization**: CL features normalized, observations standardized
4. **Dataset Creation**: Episode boundaries preserved for recurrent training
5. **Batching**: Episodes padded and collated for GPU computation

## Usage

Data loading is orchestrated by platform/runner.py which calls appropriate loader based on dataset name. Preprocessed data flows to optimizers (platform/optimizers/) for training.
