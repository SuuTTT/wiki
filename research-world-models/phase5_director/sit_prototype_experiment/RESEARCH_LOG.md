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

## 🔬 Iteration 4-5: Visual Abstraction Discovery
**Hypothesis**: The discovered SIT modules will cluster states that are easy to traverse within, while boundaries will occur at "bottlenecks" (like the narrow path on the grid).

### 1. Configuration
- **Visuals**: Matplotlib Animation + Heatmaps.

### 2. Execution & Results
- **Artifacts**:
    - Viz: [sit_discovery_evolution.gif](sit_discovery_evolution.gif)
    - Map: [original_map.png](original_map.png)

### 3. Insights
- The SIT modules successfully "cut" the map at the hole boundaries. 
- **Critical Discovery**: As the agent explores more of the map, the SIT modules shift from high-granularity (many small clusters) to stable macro-modules (2 large clusters), visually proving "Structural Entropy" reduction.
