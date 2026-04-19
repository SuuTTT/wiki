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
- **Command**: `python3 ppo_lstm_zoo.py --total-timesteps 4000000 --num-envs 8 --num-steps 128 --cuda --capture-video --save-model`
- **Status**: ⚠️ Partial Success / Diverged
- **Key Metrics**:
    - Final Reward: `~300 (at 300k steps)`
    - Explained Variance: `~0.4 (Observed)`
    - SPS: `~85`
- **Artifacts**:
    - TensorBoard: `runs/CarRacing-v3__ppo_lstm_zoo__1__1776602322/`
    - Checkpoint: `runs/[RUN_ID]/ppo_lstm_zoo.cleanrl_model`
    - Videos: `/workspace/videos/[RUN_ID]/`

### 3. Insights & Observations
- **Observation**: Reward increased to ~300 by 150k steps but then fluctuated heavily between -100 and +500.
- **Problem**: 
    - **GPU Utilization**: 0% reported because the "Nature CNN" is very small, and bottleneck is likely environment stepping/CPU overhead in `SyncVectorEnv`. 
    - **Learning Rate**: 3e-4 might be too aggressive once the agent learns basic driving, causing the observed oscillations.
    - **Entropy**: Entropy might be collapsing too fast, leading to deterministic but suboptimal policies (spinning out).
- **Action taken**: 
    - Lowered learning rate to `1e-4` for higher stability.
    - Increased `ent_coef` to `0.01` to prevent premature convergence.
    - Switched to `AsyncVectorEnv` to reduce CPU bottleneck and potentially increase SPS.
- **Hardware**: GPU driver version 12020 causes some overhead; running on CPU-dominant bottleneck.

---

## 📈 Aggregated Benchmark Table
| Exp ID | Scenario | Mean Reward | EV (Final) | Stability | Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CR-v3-PPO | Continuous | ~154 | < 0.3 | Low | [Zoo Baseline](wiki/research-world-models/phase0-car-racing-benchmark./existing.md) |
| CR-v3-LSTM | Continuous | **~862** (Target) | > 0.8 | High | [This Experiment](#) |
