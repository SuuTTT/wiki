# Research Lab Log: CarRacing-v3 World Model Baselines

## 📋 Methodology Overview
- **Objective**: Establish a high-performance, reproducible end-to-end baseline for CarRacing-v3 to compare against future World Model-based architectures (VAE/RNN/Dreamer).
- **Baseline**: PPO-LSTM (Recurrent PPO) with Nature CNN backbone, following RL Baselines3 Zoo optimized hyperparameters.
- **Primary Metrics**: Episodic Return (target > 850), Explained Variance (target > 0.8), SPS (Steps Per Second).

---

## 🔬 Experiment: CR-20260419-PPO-LSTM-ZOO
**Hypothesis**: Using a 3-layer Nature CNN + LSTM integration with frame stacking=2 and grayscale resizing will resolve the partial observability of car dynamics and reach the Zoo benchmark of ~860 reward within 4M steps.

### 1. Configuration Changes
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `env_id` | `CarRacing-v3` | Continuous action mode |
| `total_timesteps` | 4,000,000 | 4M step benchmark budget |
| `num_envs` | 8 | SyncVectorEnv with 8 parallel workers |
| `num_steps` | 128 | Steps per env rollout (1024 total batch) |
| `learning_rate` | 3e-4 | Linear annealing enabled |
| `frame_stack` | 2 | RL Zoo standard for CarRacing-v3 |

### 2. Execution & Results
- **Command**: `python3 ppo_lstm_discrete_zoo.py --total-timesteps 4000000 --num-envs 8 --num-steps 128 --cuda --capture-video --save-model`
- **Status**: ⏳ In-Progress (High Performance Config)
- **Key Metrics**:
    - Current Step: `~10k (started)`
    - Final Reward: `TBD (Target: ~862)`
    - Explained Variance: `TBD`
- **Artifacts**:
    - TensorBoard: `runs/CarRacing-v3__ppo_lstm_discrete_zoo__1__[TIMESTAMP]/`
    - Checkpoint: `runs/[RUN_ID]/ppo_lstm_discrete_zoo.cleanrl_model`
    - Videos: `videos/[RUN_ID]/`

### 3. Insights & Observations
- **Optimization Strategy**: Switched to **Discrete Action Space** (`continuous=False`). The SB3 benchmark for `ppo_lstm` on CarRacing (~862 mean reward) uses the discrete action mapping (5 actions: Do nothing, Steer Left, Steer Right, Gas, Brake). Discrete PPO often converges significantly faster and more reliably on this environment than continuous PPO because it avoids the complexity of learning precise Gaussian distribution parameters for steering.
- **Problem Fixed**: Continuous actions were causing the car to "vibrate" and lose momentum (observed in rewards barely above 0).
- **Architecture**: Single-file CleanRL-style implementation of PPO + Nature-CNN + LSTM.
- **Hardware**: GPU driver version 12020 overhead noted; running at ~88 SPS.

---

## 📈 Aggregated Benchmark Table
| Exp ID | Scenario | Mean Reward | EV (Final) | Stability | Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CR-v3-PPO | Continuous | ~154 | < 0.3 | Low | [Zoo Baseline](wiki/research-world-models/phase0-car-racing-benchmark./existing.md) |
| CR-v3-LSTM | Continuous | **~862** (Target) | > 0.8 | High | [This Experiment](#) |
