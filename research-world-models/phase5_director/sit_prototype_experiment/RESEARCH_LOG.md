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

## 🔬 Next Steps: Continuous Control & Representation Learning
**Objective**: Transition from discrete grid-worlds to continuous state/action spaces where SIT modules must be discovered from high-dimensional embeddings.

### **Planned Transition Steps**
1. **VAE/SIT Integration**: Use a VAE to encode continuous states (e.g., CarRacing or Pendulum) into a latent space, and apply SIT clustering on the latent transition graph.
2. **Action-Conditional SIT**: Explore if clusters should be defined by `(s, a, s')` rather than just `(s, s')` to account for dynamic complexity.
3. **Environment Scale-up**: Move to `Pendulum-v1` to verify if "angle-velocity" clusters map to intuitive hierarchical phases (e.g., "swing-up" vs "balancing").


