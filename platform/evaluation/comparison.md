# Behavioral Comparison Evaluation

Functions for comparing trained models against human behavior by running matched environment episodes.

## What it's for

Evaluates how well models replicate human behavior by running them on the same environment episodes (using episode seeds) and computing return statistics. Used during optimization to track behavioral alignment.

## What it contains

### Evaluation Functions
- `evaluate_progression_recurrent()` - Runs recurrent model on matched human episodes and computes percentage difference in returns
- `create_episode_list()` - Converts flat data with boundaries into episode dictionaries

## Key Details

The evaluation uses episode seeds from human data to create matched environment conditions, then runs the model's policy to collect returns. Percentage difference is computed as (model_return - human_return) / |human_return| * 100, providing a scale-invariant measure of behavioral similarity. For recurrent models, hidden states are properly initialized and maintained across timesteps. The function supports optional continual learning features (session/run IDs) that get appended to observations. Returns mean and standard deviation of percentage differences across episodes, plus raw model returns. Used by platform/optimizers/ modules during training to periodically evaluate behavioral alignment.
