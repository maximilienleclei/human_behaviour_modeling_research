# Human Behavioral Data Analysis

Analyzes and visualizes human gameplay JSON files with session statistics and training progression plots.

Contains parse_analyze_and_plot() which processes 8 data files (2 subjects × 4 environments), computes total steps/sessions/runs using 30-minute session threshold (matching preprocessing.py), and creates 4×2 subplot grid with dual y-axis plots showing returns (blue) and episode lengths (orange) with session boundaries (gray dashed). Prints statistical summary table. Standalone analysis tool not integrated into training pipeline.
