# Human Data Collection Tool

Interactive GUI for collecting human gameplay data from Gymnasium environments with keyboard input.

Uses gymnasium.utils.play with PausedSeededWrapper (pauses between episodes, increments seeds, captures timestamps) and save_data_callback() to record observations, actions, rewards, timestamps, and seeds to JSON. GAME_CONFIGS maps games to environment IDs, keyboard mappings, FPS, and help text for CartPole, MountainCar, Acrobot, and LunarLander. Seed tracking enables deterministic replay. Saves data matching naming convention used by loaders.py (e.g., sub01_data_cartpole.json).
