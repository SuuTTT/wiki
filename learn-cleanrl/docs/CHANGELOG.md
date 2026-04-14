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

### Fixed (Round 2)
- **Tensorboard Vertical X=0 Axis Collapse**: Injected dynamic `tracker.global_step = global_step` hooks directly into the core iterations of all PPO algorithms. Solved the bug where `RLTracker` plotted episodic metrics onto a single frame timestamp indefinitely.
- **PPO Continuous Plateau local minimum (-500)**: Introduced native Continuous Gymnasium normalizations into `make_env` (`ClipAction`, `NormalizeObservation`, `TransformObservation`, `NormalizeReward`, `TransformReward`). Solved the mathematical rounding bugs scaling down continuous environments.
- **Rainbow Missing Evaluation Metrics**: Updated the `rainbow_tutorial.py` logging loop to reconstruct the probability distribution (PMF array) against the `.support` bins, actively deriving scalar Q-Values so it is directly comparable to DQN evaluations. Added `tracker.log_sps()` metric output natively into the trace.
- **Tracker Injection Exceptions**: Re-wrote syntax boundaries in `ppo_envpool_tutorial.py` and `ppo_selfplay_tutorial.py` to prevent `Arg` object crashes (`AttributeError` targeting `exp_name`), successfully initiating tracking folders (`benchmark/ppo_selfplay`, `benchmark/ppo_envpool`). 


### Fixed (Round 3)
- **Multi-Agent Metric Isolation**: Repaired `ppo_selfplay_tutorial.py` only logging at `<timestamp>` 512k. The total `args.batch_size` was blindly calculated before the actual `envs.num_envs` instantiation. It is now accurately mapping metrics natively to prevent frame overflow processing unrecorded iterations.
- **Continuous PPO Plateau (-1000)**: Repaired `ppo_continuous_tutorial.py`. The `Pendulum-v1` environment's native bounds `[-2.0, 2.0]` were mathematically misaligned with the untrained architecture's zero-centered Gaussian outputs `N(0, 1)`, destroying initialization gradients. Wrapped natively with `RescaleAction(env, -1.0, 1.0)`, mapping standard deviation boundaries optimally across continuous regimes alongside optimal shorter horizon `gamma=0.9`.

