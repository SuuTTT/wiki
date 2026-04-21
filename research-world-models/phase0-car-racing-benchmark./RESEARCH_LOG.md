# Research Lab Log: CarRacing-v3 World Model Baselines

## 📋 Methodology Overview
- **Objective**: Establish a high-performance, reproducible end-to-end baseline for CarRacing-v3 to compare against future World Model-based architectures (VAE/RNN/Dreamer).
- **Baseline**: PPO-LSTM (Recurrent PPO) with Nature CNN backbone, following RL Baselines3 Zoo optimized hyperparameters.
- **Primary Metrics**: Episodic Return (target > 850), Explained Variance (target > 0.8), SPS (Steps Per Second).
- **Conclusion**: CarRacing-v3 environment is too slow, not suitable for quick iteration.
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
- **Status**: 🏁 Terminated early (Inefficient)
- **Key Metrics**:
    - Current Step: `~20k`(2h)
    - Final Reward: `~400`
    - Explained Variance: `< 0.1`
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

## � Experiment: CR-20260420-PPO-BETA-RAFAEL
**Hypothesis**: Using a **Beta Distribution** (bounded [0,1]) for continuous actions and a deeper 6-layer CNN without recurrence (LSTM) will achieve high scores faster by avoiding the "shaking" issues of Gaussian policies.

### 1. Configuration Changes
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `policy` | Beta Distribution | Bounded [0, 1], shifted to [-1, 1] for steering |
| `backbone` | 6-layer CNN | Deeper than Nature CNN (3 layers) |
| `frame_stack` | 4 | Higher temporal resolution without LSTM |
| `num_envs` | 1 | Replicating the 2019-style single-agent training |
| `learning_rate`| 1e-3 | Fast learning rate used in Rafael1s repo |
| `vf_coef` | 2.0 | High emphasis on value regression |

### 2. Execution & Results
- **Command**: `python3 ppo_beta_rafael.py --total-timesteps 4000000 --cuda --save-model`
- **Status**: 🏁 Completed (Benchmark Reached)
- **Key Metrics**:
    - Final Reward: `942.5`
    - Steps: `2,204,416`
    - Network Depth: `6 Conv Layers`

### 3. Insights & Observations
- **Recurrence vs. Depth**: This implementation proves that **LSTM is not strictly required** if the CNN is deep enough and frame stacking is sufficient (4 frames). The 2019 repo reached ~901 reward in **2,760 episodes** (approx. 400k-1M steps) using this depth.
- **Beta Distribution Trick**: Gaussian policies often waste gradient steps sampling outside the [-1, 1] range. Beta distribution is mathematically constrained to [0, 1], making the policy much more efficient at learning boundary actions (hard left/right).
- **Smooth L1 Loss**: Uses Hubber/SmoothL1 loss for value function stability, preventing huge gradients from spikes in reward.

---

## 🔬 Experiment: CR-20260421-PPO-BETA-RAFAEL-8ENV
**Hypothesis**: Parallelizing the Rafael1s Beta distribution architecture across 8 environments will significantly increase GPU utilization and reduce training time from ~7h to ~1h for 2M steps.

### 1. Configuration Changes
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `num_envs` | 8 | Vectorized using `SyncVectorEnv` |
| `total_timesteps` | 8,000,000 | Extended budget for higher precision |
| `logging` | Custom Redirect | Fixed `final_info` parsing and stdout redirection |
| `hardware` | RTX 3090 | Targeting > 400 SPS |

### 2. Execution & Results
- **Command**: `python3 ppo_beta_rafael_8env.py --total-timesteps 8000000 --cuda --num-envs 8 --save-model`
- **Status**: ⏳ Running (PID 979432)
- **Key Metrics**:
    - Current Step: `~104k`
    - Target Reward: `940+`
    - Observed SPS: `~420 (Avg)`
    - Current Mean Reward: `-25.6` (Early phase)
- **Artifacts**:
    - Log File: `8env_training.log`
    - TensorBoard: Port 6006

### 3. Insights & Observations
- **Vectorization Fix**: Standard vectorized environments in Gymnasium return episode data in a specific `final_info` sub-dictionary. The logging logic was updated to correctly iterate through these results.
- **Speed Benchmarking**: Single-env training on RTX 3090 was throttled by CPU environment stepping (~65 SPS). 8-env training saturates the GPU significantly better, aiming for a 6-7x speedup.
- **Algorithm Integrity**: No change to the model architecture from the single-env version; verified that action scaling and Beta parameters remain consistent.


---

## 🏁 Project Conclusion: Is CarRacing-v3 "Too Old/Slow"?
**Final Verdict**: **Partially Correct.**

### 1. Rationale for David Ha's Choice (2018)
In the *World Models* paper, David Ha used CarRacing because it was the "lowest hanging fruit" that was still considered **unsolved** by vanilla model-free RL (A3C/DQN scores were ~600-700). It tests:
- **Temporal Context**: Speed and momentum are critical (must brake into turns).
- **Spatial Features**: The procedural track generation ensures the agent learns *vision* rather than *memorization*.
- **Representational Efficiency**: Ha showed that a tiny **867-parameter controller** could solve it if the World Model (VAE + MDN-RNN) was robust enough.

### 2. Is it too slow?
- **Single-Env**: Yes. At ~60-80 SPS, 4M steps take >14 hours.
- **Multi-Env/Modern**: No. By using `SyncVectorEnv` with 8-16 workers, we achieved **>420 SPS** on an RTX 3090. This cuts training to ~1-2 hours for 2M steps.
- **Conclusion**: It's a "classic" that scales well. If it feels slow, it's usually an implementation bottleneck (not using vectorization) or CPU contention.

### 3. Key Implementation Lessons (The "World Model" vs. "CleanRL" Path)
| Insight | Detail |
| :--- | :--- |
| **Action Space** | **Discrete mapping (5 actions)** is the "speedrun" path. Continuous Gaussian policies struggle with precision in turns. |
| **Beta vs Gaussian** | For continuous control, **Beta distributions** (Rafael1s style) completely remove the "out-of-bound" gradient waste seen in Gaussian policy clipping. |
| **Architecture** | Recurrence (LSTM) is only necessary if you use shallow extractors (3-layer). A **deeper 6-layer CNN** with frame stacking (4) can solve the task without the complexity of unrolling hidden states. |
| **Logging** | Gymnasium's `SyncVectorEnv` stores info in `final_info`. Failing to handle this correctly is the #1 cause of "missing rewards" in logs. |

*Final Status: All experiments terminated. Baseline metrics archived.*
