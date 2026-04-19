# SIT-Director Experiment Report
*A Unified Framework for Abstract World Models and Hierarchical Planning*

## 1. Experiment Overview
This experiment evaluates the **SIT-Director** prototype on a 4x4 GridWorld navigation task. The core objective is to demonstrate that **Structural Information Theory (SIT)** can discover optimal macro-state abstractions (clusters) that serve as hierarchical goals for a Manager-Worker architecture.

### Key Metrics
- **Extrinsic Reward**: Success rate in reaching the environment goal.
- **SIT Clusters**: The number of discovered macro-states in the state transition graph.
- **Abstract Topology**: The spatial distribution of discovered clusters.

---

## 2. Quantitative Results

|   Episode |   Avg Reward |   SIT Clusters |
|----------:|-------------:|-----------:|
|       100 |          0.0 |          5 |
|       300 |          0.0 |          5 |
|       500 |          0.2 |          6 |
|       700 |          0.6 |          3 |
|       900 |          0.8 |          2 |
|      1100 |          1.0 |          2 |
|      1300 |          1.0 |          2 |
|      1500 |          1.0 |          2 |

### Performance Visualization
The following summary shows the agent's progress. As the SIT optimization compresses the state space into fewer, more meaningful macro-states (clusters), the success rate (Avg Reward) increases drastically.

**[Figure 1: Training Summary](sit_experiment_summary.png)**
*(Reward vs. Episode and Cluster Count vs. Episode)*

---

## 3. Abstract Topology Visualization
The SIT process identifies "natural joints" in the environment. In the figure below, the 4x4 grid is partitioned into discovered modules. Boundary crossings between these modules trigger new goal-setting for the Manager.

**[Figure 2: Abstract Topology Discovery](sit_abstract_visualization.png)**
*(Heatmap of discovered state clusters in the 4x4 grid)*

---

## 4. Conclusion
The experiment confirms that SIT-Director can:
1.  **Discover Structure**: Naturally partition the environment using transition graph geometry.
2.  **Coordinate Hierarchy**: Leverage discovered clusters as subgoals for stable Manager-Worker coordination.
3.  **Converge to Optimality**: Reach a 100% success rate by reducing the planning space from 16 states to 2-3 macro-modules.
