# Learn CleanRL

This repository provides customized, high-performance implementations of baseline Reinforcement Learning algorithms (like PPO, SAC, DQN, Rainbow, etc.), originally based on [CleanRL](https://github.com/vwxyzjn/cleanrl). 

## Environment

This project is tailored for **PyTorch environments on Vast.ai** (or equivalent cloud compute platforms) leveraging GPU acceleration for deep reinforcement learning.

## Fixing CleanRL's "Bit Rot"
Original CleanRL is highly optimized but suffers from "bit rot" due to heavily shifting dependencies in the RL ecosystem. This project addresses these regressions by:
1. **Gymnasium API Modernization**: Fully updated to support Gymnasium v0.26+, migrating away from deprecated `info["final_info"]` logic to the modern batched `_episode` array index parsers to properly capture terminal trajectory rewards.
2. **Centralized Logging Structure**: Introduced `RLTracker` (see `cleanrl_utils/logger.py`) to handle PyTorch `SummaryWriter` metric saving, model checkpointing (`.pth`), and structured output folder logic (`benchmark/<alg>/YYYY-MM-DD/`).
3. **Headless Cloud Rendering**: Native headless setups tricking `pygame` via `SDL_VIDEODRIVER="dummy"` to prevent crashing on Vast.ai virtual containers when rendering video traces for evaluation.
4. **Environment Enhancements**: Refactored C++ `envpool` integrations to initialize correct device placements, ensuring massive batched environments run seamlessly on GPU.

## Installation

You can install all dependencies via the provided `requirements.txt` file (preferably in an activated virtual environment or conda environment):

```bash
pip install -r requirements.txt
```

*(Note: If using `envpool` and `pettingzoo`, ensure your system has appropriate C++ build dependencies installed).*

## Launch Documentation

### Running Individual Scripts
You can easily train a single algorithm from the bash terminal. Just run the script with `python`:

```bash
# Example: Run standard PPO
python ppo_tutorial.py --total-timesteps 2048

# Example: Run High-FPS C++ Envpool PPO
python ppo_envpool_tutorial.py

# Example: Run Multi-Agent Self-Play 
python ppo_selfplay_tutorial.py
```

### Running the Benchmark Array
To stress-test all variations and populate the `benchmark/` tracking directories concurrently, run the bash array script:

```bash
chmod +x run_all_performance.sh
./run_all_performance.sh
```

This launches all tutorial scripts in the background (using `nohup`) and suppresses terminal blockages. To track their output, check their individual generated logs.
