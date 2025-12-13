# Data Preprocessing for Episodes and Continual Learning

Functions for computing temporal structure (session/run IDs), normalizing continual learning features, and creating episode-based datasets for recurrent model training.

## What it's for

Transforms raw behavioral data into structured formats suitable for training recurrent models. Handles temporal segmentation (sessions and runs) and creates PyTorch datasets that preserve episode structure for proper hidden state handling.

## What it contains

### Temporal Structure
- `compute_session_run_ids()` - Computes session and run IDs from episode timestamps (30-minute threshold for new sessions)
- `normalize_session_run_features()` - Normalizes session/run IDs to [0,1] range for use as continual learning features

### Dataset Classes
- `EpisodeDataset` - PyTorch Dataset that wraps episodes for recurrent training
- `episode_collate_fn()` - Custom collate function that pads episodes to batch them together

## Key Details

Sessions represent distinct periods of human play separated by at least 30 minutes, while runs are sequential episodes within a session. This temporal structure can be used as continual learning features to help models adapt to changing human behavior over time. The EpisodeDataset preserves episode boundaries (required for recurrent models to properly reset hidden states between episodes) and the collate function handles variable-length episodes via padding. Used by platform/optimizers/sgd.py for batched recurrent training.
