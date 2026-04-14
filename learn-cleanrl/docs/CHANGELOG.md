# Changelog

All notable changes to the CleanRL Tutorial and Benchmark project will be documented in this file.

## [Unreleased] - 2026-04-14

### Added
- **EnvPool Integration**: Created `ppo_envpool_tutorial.py` and `EnvPool.md` to demonstrate high-throughput zero-copy C++ batched environment execution.
- **RLTracker Utility**: Implemented a standalone centralized `cleanrl_utils/logger.py` class that manages TensorBoard event logging and automated PyTorch `.pth` checkpointing.
- **Benchmark Data Standardization**: Upgraded the `RLTracker` to format and route all metric data directly to `benchmark/<algorithm>/YYYY-MM-DD-<timestamp>`.
- **Render Guidance**: Added `render_envpool.md` detailing the "Train Fast, Evaluate Slow" methodology for headless/high-fps generated models.
- **Curriculum Expansion**: Added "Part 7: Hierarchical Planning, Abstraction, and RL" to `curriculum.md`.
- **Concurrency Test Suite**: Generated `run_all_performance.sh` to seamlessly launch algorithms natively in the background via `nohup`.

### Fixed
- **Multi-Agent Compilation Error**: Stripped out `multi-agent-ale-py` (which fails on Python 3.12) and transitioned `ppo_selfplay_tutorial.py` entirely to PettingZoo's native Python `pistonball_v6`.
- **Headless PyGame Crash**: Injected `os.environ["SDL_VIDEODRIVER"] = "dummy"` to prevent X11 server runtime errors in the Vector environments.
- **Boolean Tensor Crash**: Resolved the `1.0 - next_done` boolean inverse RuntimeError by explicitly mapping `done` arrays to `torch.float32`.
- **Gymnasium API Deprecation**: Swept across all baseline tutorial scripts to replace the deprecated `.info["final_info"]` logic with the modern `"_episode"` boolean array dictionaries. TensorBoard logging is now perfectly aligned.
- **Missing PPO Metrics**: Refactored PPO and PPG algorithm scripts to actively append the `value_loss`, `policy_loss`, `entropy`, and `SPS` to the logging tracker.
- **Git Exclusions**: Updated `.gitignore` to catch local `tfevents`, video mp4s, caches, and test logs.
