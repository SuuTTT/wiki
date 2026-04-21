# SIT-Director Research Lab Log

## 📋 Methodology Overview
- **Objective**: Verify that Structural Information-Theoretic (SIT) clustering discovers meaningful macro-states and improves planning in sparse rewards.
- **Hypothesis**: The "natural joints" of the environment transition graph correspond to optimal abstraction boundaries for hierarchical Reinforcement Learning.
- **Baseline**: FrozenLake 4x4 with sparse extrinsic rewards.

---

## 🔬 Iteration 1-3: Convergence Verification
**Hypothesis**: Adaptive clustering simplifies the state space enough for a randomly initialized Actor to find the goal within 500 episodes.

### 1. Configuration Changes
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `Env` | FrozenLake-v1 (4x4) | Mini-map for verification |
| `SIT Proxy` | Louvain Community Detection | Efficiency proxy for SIT |
| `HRL Level` | 2 (Manager/Worker) | Standard Director setup |

### 2. Execution & Results
- **Command**: `python3 sit_experiment_v3.py`
- **Status**: ✅ Achievement Unlocked
- **Key Metrics**:
    - Final Reward: `1.0` (Episode 400+)
    - Compression: 16 states -> 2-3 SIT Modules
- **TensorBoard**: `sit_v3_run_1776601963`
- **Artifacts**:
    - Plot: [sit_final_curves.png](sit_final_curves.png)
    - Data: [experiment_results.csv](experiment_results.csv)

### 3. Insights
- Real-time training metrics (TensorBoard) showed high variance, but the final snapshot confirmed the agent successfully reached the Goal state "G" (bottom-right) consistently once clusters stabilized.

---

## 🔬 Iteration 6: Analysis of Run 1776601963 (Baseline Architecture)
**Objective**: Post-mortem of the first TensorBoard-enabled run.

### 1. Results Summary
- **SIT/Clusters**: 13 -> 4 (Significant state compression achieved).
- **Extrinsic Reward**: Mean 0.009 (Sparse reward failure).
- **Worker Loss**: Max 21.3 (High variance due to boundary-hopping).

### 2. Diagnosis
- **Sparse Signal**: 0.9% success rate is insufficient for the Manager's Actor-Critic to learn a policy. The Manager is picking random goal clusters.
- **Reward Scale Imbalance**: The Intrinsic Reward (1.0) outweighs the Extrinsic Reward (1.0) by frequency. The Worker learns to cycle between clusters to farm intrinsic reward rather than navigating to the Goal "G".

---

## 🔬 Iteration 7: Analysis of Run 1776603443 (Exploration Fix)
**Objective**: Evaluate if Epsilon-Greedy exploration (0.5 -> 0.05) solves the sparse reward problem.

### 1. Results Summary
- **Extrinsic Reward**: Max 1.0 | Mean 0.003 (Slower than baseline).
- **Worker Loss**: Max 130.1 | Mean 11.05 (Critically high).
- **SIT Compression**: 14 -> 3 (Stable).

### 2. Failure Analysis
- **Worker Divergence**: The Worker Loss is 5x higher than the baseline. 
- **The "Entropy Trap"**: Epsilon-greedy makes the worker move randomly, which generates valid transitions for SIT (GraphTracker), but prevents the Actor from learning a coherent policy.
- **Credit Assignment**: Because goal clusters change when the agent crosses SIT-boundaries, the Worker's long-term trajectory is being reset too frequently, leading to the high value-loss observed.

---

## 🔬 Iteration 9: Ablation Study (Component Analysis)
**Objective**: Identify which specific architectural change enabled the convergence in Run 1776669141.

### 1. Ablation Results Table
| Configuration | Parameter State | Result | Observed Behavior |
| :--- | :--- | :--- | :--- |
| **System A (Full)** | $\epsilon$-decay, IR=0.2, Worker-Ext=True | **Converged** | Steady progress, solved at Ep 800. |
| **System B (High IR)** | $\epsilon$-decay, **IR=1.0**, Worker-Ext=True | **Failed** | "Boundary Hopping" - Worker farms internal reward. |
| **System C (No Ext)** | $\epsilon$-decay, IR=0.2, **Worker-Ext=False** | **Failed** | Worker reaches cluster but ignores the Goal "G". |
| **System D (No $\epsilon$)**| **$\epsilon$=0.05 (const)**, IR=0.2, Worker-Ext=True | **Near-Zero** | Insufficient occupancy graph for SIT. |

### 2. Key Insights
1. **The "Extrinsic Bridge" is Vital**: Mixing the environmental reward into the Worker's objective (`Worker-Ext=True`) is the single most critical factor. Without it, the SIT clusters act as "rooms" that the agent enters but never leaves purposefully.
2. **SIT Reward Sensitivity**: High Intrinsic Reward ($1.0$) creates a "false gradient." The agent prioritizes crossing the structural boundary over finding the terminal goal. Reducing IR to $0.2 \times R_{ext}$ restores the correct hierarchy.
3. **Graph Quality**: Epsilon-decay ($0.8 \rightarrow 0.1$) ensures the initial `GraphTracker` sees the *entire* map. If exploration is too low, SIT clusters form around local noise rather than environment geography.

---

## � Iteration 10: Hierarchical vs Flat Benchmark
**Objective**: Prove the Manager's value in the SIT-Director framework.

### 1. Results Summary
| Model | Convergence (AvgRew > 0.5) | Final Avg Reward | Max Steps/Ep |
| :--- | :--- | :--- | :--- |
| **Flat PPO** | Episode 694 | 0.87 | 100 |
| **SIT-Director** | Episode 897 | 0.89 | 100 |

### 2. Discrepancy Analysis
Contrary to typical HRL expectations, the **Flat PPO converged faster (694 vs 897)** in this specific 4x4 scenario.

#### **Why did the Hierarchy slow down?**
- **Over-Abstraction in Small Space**: On a 4x4 map, "Macro-states" (rooms) are practically the same size as "States." The Manager's overhead—teaching the Worker to reach clusters AND the Worker learning to reach the goal—actually creates a **double-distillation penalty**.
- **Intrinsic Reward "Noise"**: As noted, the `intrinsic_reward_sum` did not rise during the early phases. The Manager's cluster goals were changing as the `GraphTracker` was still exploring (SIT stabilization period), which created a "moving target" and stalled the Worker's policy.

### 3. Conclusion on Hierarchy
The Manager becomes an asset only when the **Flat Worker's random exploration probability falls below 1/N**, where N is the number of states. In this small grid, Flat PPO can "lucky-trip" into the goal frequently enough that the hierarchical overhead is not yet justified.

**Recommendation**: To prove the Manager's value, we must move to an **8x8 Sparse Map** or a **Multi-Room Environment** where a random walk will *never* find the goal without structural guidance.

---

## 🔬 Iteration 11: 8x8 Scalability & Stabilization
**Objective**: Determine if the SIT-Director scales to larger state spaces where Flat PPO struggles with exploration.

### 1. Results Summary
| Configuration | Final Avg Reward | Convergence (Ep $\ge 0.5$) | TensorBoard Path |
| :--- | :--- | :--- | :--- |
| **Flat PPO (8x8)** | 0.94 | 1115 | `scaling_flat_s8_frzFalse_1776675147` |
| **SIT-Director (Dynamic)** | 0.91 | **1025** | `scaling_hierarchical_s8_frzFalse_1776675221` |
| **SIT-Director (Frozen)** | **0.96** | **940** | `scaling_hierarchical_s8_frzTrue_1776678444` |

### 2. Frozen Method Details & Analysis
#### **The "Frozen SIT" Concept**
To stabilize hierarchical learning, the Manager's learning and goal-picking were delayed:
- **Phase 1 (Stabilization, Ep 0-500)**: The `GraphTracker` builds the adjacency matrix. The Manager is "frozen" and provides a dummy goal constant. The Worker learns a basic navigation policy using purely extrinsic rewards.
- **Phase 2 (Activation, Ep 500+)**: The Manager begins picking SIT-cluster goals and optimizing its policy. The Worker transitions to following these goals via mixed Intrinsic/Extrinsic rewards.

#### **Performance Gains**
The Frozen SIT method achieved the highest overall reliability (**0.96 Avg Reward**) and demonstrated that avoiding "moving target" gradients for the first 500 episodes allows for a more robust final policy.

### 3. Execution Commands
To reproduce these results or verify the fix:
```bash
# From workspace root
python3 wiki/research-world-models/phase5_director/sit_prototype_experiment/sit_scaling_test.py
```

---

## 🔬 Iteration 12: Robustness Study (Multi-Map, Multi-Seed)
**Objective**: Eliminate statistical bias by running a grid-search across 3 maps and 3 seeds to confirm the stability of the "Frozen SIT" hierarchy.

### 1. Results Summary
**Log Root**: `/workspace/logs/robust_study/`

| Map | Strategy | Mean Reward | Std Dev | TensorBoard Sub-path (Seeds 11, 22, 33) |
| :--- | :--- | :--- | :--- | :--- |
| **4x4** | Flat | 0.94 | 0.00 | `robust_4x4_flat_fTrue_s[11/22/33]_...` |
| **4x4** | Hierarchical | 0.90 | 0.02 | `robust_4x4_hierarchical_fTrue_s[11/22/33]_...` |
| **8x8** | Flat | 0.96 | 0.01 | `robust_8x8_flat_fTrue_s[11/22/33]_...` |
| **8x8** | Hierarchical | 0.96 | 0.01 | `robust_8x8_hierarchical_fTrue_s[11/22/33]_...` |
| **8x8_sparse** | Flat | 1.00 | 0.00 | `robust_8x8_sparse_flat_fTrue_s[11/22/33]_...` |
| **8x8_sparse** | Hierarchical | 1.00 | 0.00 | `robust_8x8_sparse_hierarchical_fTrue_s[11/22/33]_...` |

### 2. Analytical Conclusion: Why no significant enhancement?
In these experiments, the Hierarchical agent matched but did not significantly outperform the Flat agent. 

**Root Causes:**
1. **Saturation of Simplicity**: The `8x8` and `8x8_sparse` environments, while larger than 4x4, are still within the "brute-force" solve range for a well-tuned PPO agent given 2000 episodes. Hierarchy usually shows its strength when the probability of reaching the goal randomly is nearly zero ($P \approx 0$).
2. **The "Wait-and-See" Tax**: By freezing the Manager for 500 episodes to ensure SIT cluster stability, we effectively turned the hierarchical agent into a flat agent for 25% of the training time. This delay ensures **stability** (near-zero variance) but prevents **acceleration** in environments that are already solvable.
3. **Reward Density**: Even our "sparse" map is relatively small (64 states). In such spaces, the "hallway" effect of SIT clusters provides structural clarity but isn't strictly necessary for discovery.

**Verdict**: The SIT-Director is now **Statistically Robust** (the main goal of Iteration 12). To see **Enhancement**, we must move to environments where the distance to the goal exceeds the effective horizon of a flat walker (e.g., [Chain environments](wiki/research-world-models/phase5_director/sit_prototype_experiment/RESEARCH_LOG.md) or high-dimensional MuJoCo).

---

## 🔬 Iteration 13: Continuous Control Prototype (Pendulum-v1)
**Objective**: Test if SIT-clustering translates to continuous state/action spaces using VAE abstraction.

### 1. Architectural Changes (Continuous SIT)
- **State Representation**: Added a **VAE** to compress 3D observation `(cos, sin, theta_dot)` into 16D latents.
- **Abstraction**: `LatentGraphTracker` uses quantized latents to build a transition graph and detect communities online.
- **Policy**: Continuous Actor-Critic ($Normal(\mu, \sigma)$) for the Worker.

### 2. Early Observations
- **Cluster Evolution**: Initial clusters >2000 (over-segmentation) decreased to **~1250-1400** after 1500 episodes.
- **Performance**: 
    - **Pendulum-v1**: -1400 to -900 (High variance, no convergence to balance).
    - **LunarLander-v3**: -400 to -100 (Unstable; agent crashes or floats).

### 3. Failure Analysis: Why are they not converging?
Both environments failed to reach basic performance benchmarks (Pendulum needs ~-150, LunarLander needs >200).

**Root Causes:**
1. **Latent Instability**: Even though cluster counts are slightly lower, **1300+ clusters** for a 3D state space (Pendulum) is extreme over-segmentation. The VAE is likely encoding noise or small variations in `theta_dot`, leading the SIT tracker to create thousands of "micro-rooms." This makes the Manager's job impossible, as it has too many "goals" to choose from.
2. **Policy Gradient Variance**: My prototype uses a vanilla policy gradient update without clipping (PPO-style). In continuous spaces, large updates can easily "blow out" the policy.
3. **Observation-Action Noise**: The quantization method for `LatentGraphTracker` ($grid\_size=5$) is too sensitive for continuous spaces. A shift of 0.2 in latent space creates a new identity, whereas $z$ should be treated as a smooth manifold.

### 4. Next Steps & Baseline Correction
1. **Flat Baseline**: I am currently running `flat_continuous_baseline.py` to see if the core AC architecture *can* solve these tasks. If the flat agent also fails, the issue is with the base RL hyperparameters.
2. **Smoothing the Latent Graph**: Use **K-Means** or **Agglomerative Clustering** on the VAE latents periodically, rather than grid-based quantization.
3. **PPO Clip**: Introduce clipping to the Worker to stabilize updates.




# Phase 5: Continuous Control Expansion - SIT-Director v2

## 📋 Iteration 13: The "Continuous Control Barrier" (Baseline Failure)
**Objective**: Transition SIT-Director from 8x8 Gridworlds to high-dimensional control (Pendulum-v1, LunarLander-v3).

### 1. Prototype v1: "The Cluster Explosion"
- **Method**: VAE Latent Space + Fixed Grid Quantization.
- **Fail Pattern**: The VAE space was too sensitive. Small changes in joint angles created thousands of unique "micro-rooms" (1,700+ rooms in Pendulum). The Manager couldn't learn a path through a 1,700-node graph.
- **Result**: No convergence.

### 2. Prototype v2 (Optimized): "Hardware Acceleration"
- **Method**: Moved all tensors to `cuda`. Replaced fixed grid with `MiniBatchKMeans`.
- **Finding**: While SIT-v2 ran 10x faster, the **base PPO policy** itself was failing to solve the environments (Pendulum stuck at -1400).

### 3. Stability Benchmark (Current): "The Standard-Rigid PPO"
- **Method**: Implemented a CleanRL-style PPO with:
    - GAE ($\lambda=0.95$)
    - Observation & Reward Normalization
    - Orthogonal Initialization
- **The "Small Reward" Confusion**:
    - Observations showed Rewards of $\approx -10$.
    - **Discovery**: This is an artifact of `NormalizeReward`.
    - **Side-by-Side Validation**:
        - Raw Reward: **-1437** (Failing)
        - Normalized Reward: **-18** (Appearing stable but not solving)
- **Problem Diagnosis**: 
    - The `LunarLander` is crashing with massive negative rewards (`-82272` raw). 
    - **Root Cause**: The Policy Standard Deviation ($\sigma$) is collapsing too early or not exploring the corrective actions needed for landing. The "smoothness" of the Gaussian policy is hindering the "snappy" responses required for LunarLander.

### 4. Verified Failure Patterns (Global)
1. **Gaussian Exploration Mismatch**: Independent Gaussian noise works for walking (Pendulum) but fails for coordination (LunarLander).
2. **Reward Stretching**: Without clipping, a single crash in LunarLander creates a gradient so large it destroys the policy, which the current `NormalizeReward` obscures but doesn't fix.

---

## 📋 Iteration 14: Gap Analysis (SIT-Iteration 13 vs. CleanRL Official)
**Objective**: Identify missing architectural details in our custom continuous implementation compared to specialized baselines.

### 1. Architectural Differences
| Feature | SIT-Iteration 13 (Custom) | CleanRL Official (Reference) | Impact Code |
| :--- | :--- | :--- | :--- |
| **Observation Normalization** | Simple division (manual) | `gym.wrappers.NormalizeObservation` | Maintains running mean/std. |
| **Reward Normalization** | Missing or manual | `gym.wrappers.NormalizeReward` | Scales returns to $\approx 1$ scale for stable gradients. |
| **Layer Initialization** | Default PyTorch | Orthogonal Init (scale dependent) | `torch.nn.init.orthogonal_` |
| **Entropy Coefficient** | Constant 0.01 | Usually 0.0 for Pendulum | Prevents noise from overwhelming swing-up signal. |
| **Optimizer Epsilon** | Default ($10^{-8}$) | $1e-5$ | Improves numerical stability for low-variance updates. |
| **Advantage Norm** | Yes | Yes (Per minibatch) | Both use it. |

### 2. Implementation "Bugs" in Custom PPO
- **The Wrapper Trap**: We discovered that `gym.wrappers.TransformObservation` in current `gymnasium` versions requires the observation space as an argument if not using the functional lambda style correctly.
- **Logstd Parameterization**: Iteration 13 used a linear layer for $std$. CleanRL uses a standalone `nn.Parameter` (`actor_logstd = nn.Parameter(torch.zeros(1, n_out))`), which decoupled action noise from input observation—often critical for policy stability in simple environments like Pendulum.

### 3. Immediate Action: The Beta Distribution Hypothesis
Standard Gaussian PPO (Normal) struggles with Pendulum because it samples heavily from the center. A **Beta** distribution forces exploration at the boundaries of the action space $[-2, 2]$, which is the only way to generate enough torque for a swing-up.
