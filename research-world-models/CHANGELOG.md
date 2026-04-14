# Changelog

## [1.0.0] - Phase 1: Representation Learning
### Added
- Created `phase1_representation_learning/01_world_models_vae_rnn.py` for joint PPO+VAE+MDN-RNN representation reinforcement learning.
- Created `phase1_representation_learning/02_ppo_baseline_cnn_rnn.py` for a pure RL comparison baseline on identical model capacity without auxiliary reconstruction losses.
- Created `phase1_representation_learning/01_render_video.py` for evaluating and rendering agent performance.
- Created `phase1_representation_learning/docs/design.md` and `docs/launch.md`.
- Isolated TensorBoard data (`benchmark/`) and Logs (`logs/`) per pure RL vs. Representation RL runs.
- Initialized `CHANGELOG.md` and `TODO.md` to plan for David Ha's True World Model reimplementation.