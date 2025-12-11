# Human Data Collection Tool

Interactive GUI-based tool for collecting human gameplay data from Gymnasium environments with keyboard input.

## What it's for

Enables human subjects to play Gymnasium environments while automatically recording observations, actions, rewards, timestamps, and episode seeds for later model training. Provides paused episode-by-episode recording with seed tracking.

## What it contains

### Custom Wrapper
- `PausedSeededWrapper` - Gym wrapper that pauses between episodes, increments seeds, and captures timestamps

### Callback Function
- `save_data_callback()` - Records observations, actions, rewards after each step and saves episodes to JSON

### Game Configurations
- `GAME_CONFIGS` - Dictionary mapping game selection to environment IDs, keyboard mappings, FPS, no-op actions, and help text

### Supported Environments
1. CartPole-v1 - Balance pole with left/right arrows
2. MountainCar-v0 - Build momentum with left/right acceleration
3. Acrobot-v1 - Swing to target with torque controls
4. LunarLander-v3 - Land safely with main and side engines

## Key Details

Uses gymnasium.utils.play for interactive keyboard-controlled gameplay with pygame rendering. The PausedSeededWrapper pauses after each episode termination, waiting for spacebar press to start next episode with incremented seed. This seed tracking enables deterministic replay via data/replay.py. Each episode records full trajectory: observations (state vectors), actions (integer action indices), rewards, timestamp (ISO format), and seed. Data saved to JSON files matching naming convention used by platform/data/loaders.py (e.g., sub01_data_cartpole.json). Keyboard mappings configured per-environment to match intuitive controls. FPS settings balance playability and human reaction time. The collected data becomes training input for behavior modeling experiments.
