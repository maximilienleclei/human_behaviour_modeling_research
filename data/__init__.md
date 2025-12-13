# Data Directory

Repository for all behavioral data files and data processing code.

## Structure

### hf_control_tasks/
HuggingFace dataset loaders for pre-trained agent trajectories.
- **loaders.py** - Load CartPole-v1 and LunarLander-v2 from HuggingFace
- Data from NathanGavenski repositories
- See `hf_control_tasks/__init__.md` for details

### simexp_control_tasks/
Human behavioral data from SimExp experiments.
- **source/** - Raw JSON data files (UNTOUCHED)
- **environments.py** - Environment configurations for simexp tasks
- **loaders.py** - Human data loader with CL features
- **preprocessing.py** - Temporal segmentation, normalization, episode datasets
- See `simexp_control_tasks/__init__.md` for details

## Purpose

This directory centralizes all data-related code and files:
- Data **files** (human behavioral data in `simexp_control_tasks/source/`)
- Data **loading** (HF loaders in `hf_control_tasks/`, simexp loaders in `simexp_control_tasks/`)
- Data **preprocessing** (simexp-specific preprocessing in `simexp_control_tasks/`)

## Import Guide

- HuggingFace data: `from data.hf_control_tasks import load_cartpole_data, load_lunarlander_data`
- SimExp data: `from data.simexp_control_tasks import load_human_data, ENV_CONFIGS`
- SimExp preprocessing: `from data.simexp_control_tasks import compute_session_run_ids, EpisodeDataset`
