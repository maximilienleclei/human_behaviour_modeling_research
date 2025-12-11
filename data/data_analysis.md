# Human Behavioral Data Analysis

Analysis and visualization script for human gameplay session data across multiple environments.

## What it's for

Analyzes human behavioral data JSON files to compute statistics (total steps, session counts, runs per session) and visualize training progression (returns, episode lengths, session boundaries) over time.

## What it contains

### Main Function
- `parse_analyze_and_plot()` - Processes multiple JSON files, computes session statistics, and creates visualization grid

### Analysis Features
- Total step counting across all episodes
- Session segmentation using 30-minute threshold (matches platform/data/preprocessing.py logic)
- Runs per session computation
- Statistical summary table printed to terminal

### Visualization
- 4×2 subplot grid for 8 data files (sub01/sub02 × 4 environments)
- Dual y-axis plots: total return (blue scatter) and episode length (orange x markers)
- Session boundary lines (gray dashed) showing temporal segmentation
- Output saved as PNG file

## Key Details

The script is designed for quick exploratory analysis of collected human data. It processes all 8 standard data files (2 subjects × 4 environments: CartPole, MountainCar, Acrobot, LunarLander). Session detection uses 30-minute gap threshold consistent with platform/data/preprocessing.py's compute_session_run_ids(). The dual-axis visualization reveals both task performance (returns) and behavioral efficiency (episode lengths) within the same plot. Session boundaries help identify temporal structure and learning phases. This script operates independently from the platform layer - it's a standalone analysis tool, not integrated into the training pipeline.
