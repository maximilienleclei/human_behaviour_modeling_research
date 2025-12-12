# Episode Replay Tool

Deterministic replay tool for visualizing recorded human gameplay episodes using environment seeds.

## What it's for

Allows deterministic replay of previously recorded human episodes by resetting environments with the same seeds. Useful for debugging data collection, verifying episode consistency, and visualizing human behavior.

## What it contains

### Configuration
- `GAME_CONFIGS` - Dictionary mapping game selections to environment IDs, filenames, and FPS settings

### Replay Logic
- Main script loads JSON data file
- Iterates through episodes, resetting environment with recorded seed
- Replays actions step-by-step with timing control
- Renders episodes in human-viewable mode

## Key Details

The replay works because Gymnasium environments are deterministic when seeded - resetting with the same seed produces identical initial conditions, and the same action sequence produces identical trajectories. FPS control maintains recorded playback speed. The tool requires the same environment configuration used during recording (data/collect.py). Useful for verifying that recorded episodes are valid and for understanding human decision-making patterns visually. Episodes are paused between replays waiting for user input (Enter to continue, 'q' to quit).
