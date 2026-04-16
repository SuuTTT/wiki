# Launch Document: Custom JAX RL on MuJoCo

This document tracks the initialization and setup for developing custom JAX-based RL algorithms and environments using **MuJoCo Playground** and **MJX**.

## 🎯 Goal
Implement a custom JAX-based RL algorithm and a custom MuJoCo environment, following the [WORKFLOW.md](../../WORKFLOW.md) best practices.

## 🛠 Prerequisites & Environment
- **Workspace**: `/workspace/wiki/learn_mujoco_playground/`
- **Source**: [MuJoCo Playground Repo](repo/)
- **Virtual Env**: `.venv_playground` (Python 3.12, JAX with CUDA12)
- **Reference Code**: `repo/learning/train_jax_ppo.py`
- **Core Dependencies**: `jax`, `mujoco`, `mujoco_mjx`, `brax`, `flax`, `optax`

## 📦 Project Structure
```
wiki/learn_mujoco_playground/
├── launchdoc.md           # This file
├── plandoc.md             # Layered execution plan
├── mujoco_playground_paper.pdf
├── logs/                  # Training logs and checkpoints
├── repo/                  # MuJoCo Playground source
└── custom/                # Custom development area
    ├── envs/              # Custom MJX environments
    ├── algos/             # Custom JAX algorithms
    └── tutorials/         # EXPLAINED.md outputs
```

## ✅ Initial Verification (Smoke Test)
- [x] MuJoCo Playground installed from source.
- [x] `jax.default_backend()` is `gpu`.
- [x] `G1JoystickFlatTerrain` assets downloaded.
- [x] Baseline PPO run on `CartpoleBalance` successful.

## � Execution Commands

### Baseline PPO (MuJoCo MJX)
Run the standard PPO implementation on DM Control Suite environments:
```bash
cd /workspace/wiki/learn_mujoco_playground/repo
# Syntax: uv run train-jax-ppo --env_name <ENV_NAME>
uv --no-config run train-jax-ppo --env_name CartpoleBalance --num_timesteps 1000000
```

### Baseline PPO (MuJoCo Warp)
To run with the Warp backend for potential speedup on supported environments:
```bash
cd /workspace/wiki/learn_mujoco_playground/repo
uv --no-config run train-jax-ppo --env_name CartpoleBalance --impl warp
```

### RSL-RL PPO (Legged Robotics optimized)
Often used for locomotion tasks like quadruped walking:
```bash
cd /workspace/wiki/learn_mujoco_playground/repo
uv --no-config run train-rsl-ppo --env_name G1JoystickFlatTerrain
```

When you run uv run, you don't need to name the environment because uv is context-aware. It automatically searches the current directory and its parent for a .venv folder. Since your repository is located at repo, and it contains a .venv folder, uv immediately finds and uses the correct interpreter.



## ⚙️ Control & Logging

### What `uv run` does
In this project, `uv run` handles the heavy lifting of environment management:
1.  **Environment Auto-Selection**: You don't need to name the environment (like `.venv_playground`) because `uv` automatically looks for a `.venv` folder in the current directory or parent directories. Since we are in `repo/`, it finds and uses the local virtual environment.
2.  **Path Resolution**: It automatically adds the `mujoco_playground` package to your `PYTHONPATH`, so `import mujoco_playground` just works.
3.  **Path**: The Python interpreter used is located at: `/workspace/wiki/learn_mujoco_playground/repo/.venv/bin/python`.

### Custom Log Directory
By default, logs are stored in `repo/logs/`. Use `--logdir` to change this:
```bash
--logdir <PATH_TO_LOGS>
```

### Enabling Dashboards
- **TensorBoard**: Add `--use_tb` to the command to save events files.
- **Weights & Biases**: Add `--use_wandb` to the command (requires `wandb login`).
## �📝 Metrics Contract (Standard JAX/RL)
Log via `RLTracker` or `tensorboard`:
- `charts/episodic_return`
- `charts/episodic_length`
- `charts/SPS`
- `losses/policy_loss`
- `losses/value_loss`
- `losses/entropy`
- `debug/grad_norm`
