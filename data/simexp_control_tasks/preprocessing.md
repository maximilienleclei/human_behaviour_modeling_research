# Data Preprocessing for Episodes and Continual Learning

Temporal structure computation and episode-based dataset creation for recurrent models.

Contains compute_session_run_ids() for extracting sessions/runs from timestamps (30min threshold), normalize_session_run_features() for scaling to [-1,1], EpisodeDataset class for preserving episode boundaries, and episode_collate_fn() for padding variable-length episodes. Sessions are distinct play periods separated by 30+ minutes; runs are sequential episodes within a session. Used by dl/optim/sgd.py for batched recurrent training with proper hidden state handling.
