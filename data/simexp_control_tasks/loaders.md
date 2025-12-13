# Human Behavioral Data Loader

Loads human gameplay JSON files with continual learning features and per-session train/test splitting.

Contains load_human_data() which reads JSON episodes from source/, computes session/run IDs from timestamps, normalizes temporal features to [-1,1], performs per-session 10% random holdout (seed 42), and optionally concatenates normalized session/run features to observations when use_cl_info=True. Returns optim/test tensors plus metadata dict with episode boundaries, session/run IDs, and test episode info for evaluation. Used by dl/optim/sgd.py and ne/optim/*.py for training on human behavioral data.
