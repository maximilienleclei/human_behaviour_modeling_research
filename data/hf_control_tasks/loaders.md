# HuggingFace Dataset Loaders

Loads pre-trained agent trajectories from NathanGavenski's HuggingFace datasets. Contains load_cartpole_data() and load_lunarlander_data(), which download datasets, convert to PyTorch tensors, shuffle, and return 90/10 train/test splits of observation-action pairs.
