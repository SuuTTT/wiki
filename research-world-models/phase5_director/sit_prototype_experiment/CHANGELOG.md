# SIT-Director Code ChangeLog

This file tracks architectural changes, bug fixes, and feature additions to the SIT-Director prototype codebase.

## [2026-04-19] - Initial Prototyping
### Added
- `sit_prototype.py`: Core implementation of GraphTracker (Louvain proxy for SIT) and 2-level HRL.
- `sit_experiment_v3.py`: Optimized training script with deterministic 4x4 Mini-Map support.
- `sit_final_experiment.py`: Integrated TensorBoard logging for metric tracking.
- `sit_viz_video.py`: Matplotlib animation script for state-space evolution.
- `sit_plot_final.py`: Visualization suite for final research plots.

### Fixed
- Fixed `RuntimeError` regarding Long vs Float dtypes in reward tensors.
- Corrected state mapping in `GraphTracker` to handle zero-indexed environment states correctly.
- Resolved dependency issues for `pandas` and `tabulate` via local pip installation.

### Refactored
- Moved all experimental artifacts into `sit_prototype_experiment/` subdirectory.
- Standardized Manager/Worker action spaces for discrete grid-world environments.
