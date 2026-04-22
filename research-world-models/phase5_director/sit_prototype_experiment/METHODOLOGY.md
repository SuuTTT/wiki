# SIT-Director Methodology and Algorithm Specification

## 0. Fundamental Theory: SIT as a Non-Parametric World Model
In this framework, we define the **World Model** not as a generative network that predicts pixels, but as a **graph-based topological abstraction** ($\mathcal{G} = \{V, E\}$).
- **Vertices ($V$):** Points of interest in the state space (discrete squares, latent clusters, or visual landmarks).
- **Edges ($E$):** Discovered reachability between vertices.
- **Hierarchy ($\mathcal{M}$):** Communities of vertices discovered by minimizing the SIT functional (Structural Entropy). This represents the "True Logic" of the map (e.g., "The Hallway," "The Left Room").

The Director uses this graph to set goals in the space of $\mathcal{M}$, effectively planning over the environment's topology rather than its raw coordinates.

---

## 1. Discrete Domain (Gridworld) Architecture
**Status**: Validated (Iterations 1-12)
**Component Breakdown**:

### A. Graph Construction (GraphTracker)
- **Input**: Discrete state $s_t$, transition $(s_t, s_{t+1})$.
- **Mechanism**: Maintains an adjacency matrix $A$ where $A_{ij}$ counts transitions from state $i$ to $j$.
- **Symmetry**: $A = (A + A^T)$ creates an undirected graph for structural analysis.

### B. State Abstraction (SIT-Clustering)
- **Objective**: Minimize the Structural Information-Theoretic (SIT) functional for partitioning.
- **Proxy**: Louvain Community Detection (optimized for modularity).
- **Update Frequency**: Every $N$ episodes (e.g., 50), re-cluster states into macro-states $\mathcal{M}$.

### C. Hierarchical Policy (The Director)
- **Manager**: 
    - **Action**: Selects a goal macro-state $g \in \mathcal{M}$.
    - **Reward**: Uses extrinsic environmental reward.
- **Worker**:
    - **Action**: Discrete movement (Up, Down, Left, Right).
    - **Reward**: $r_{total} = r_{ext} + \lambda \cdot \mathcal{I}(s_{next} \in g)$.
    - $\mathcal{I}$ is the indicator function for reaching the Director's goal cluster.

### D. The "Frozen SIT" Trick (Stabilization)
- **Phase 1 (Stabilization)**: Manager is "frozen". Worker learns purely from $r_{ext}$ while `GraphTracker` populates $A$.
- **Phase 2 (Activation)**: SIT clusters $\mathcal{M}$ are computed and "frozen" for a period. Manager begins learning to sequence these clusters.
- **Example (4x4 Map)**: 
    - States 0-15.
    - SIT identifies Clusters: $G_0 = \{0,1,4,5\}$ (Top-Left), $G_1 = \{2,3,6,7\}$ (Top-Right), etc.
    - Manager picks $G_0 \rightarrow G_1$. Worker only needs to learn "Exit $G_0$ to the Right".

---

## 2. Continuous Domain Architecture (Current v2)
**Status**: Prototyping (Iterations 13-16)
**Choice of Backbone**: **PPO (Stable)** or **TD-MPC2 (Performance)**

### A. Latent Abstraction and SIT Loss
The core of the continuous world model is the integration of **Structural Information Theory** into the latent dynamics. 

1.  **SIT Graph Construction**: 
    - Vertices $V$ are defined as latent states $z_{0:T}$ from a rollout.
    - An **Adjacency Matrix** $A$ is computed via a **Heat Kernel** (RBF) on pairwise Euclidean distances:
      $$A_{ij} = \exp\left(-\frac{\|z_i - z_j\|^2}{2\sigma^2}\right)$$
    - This creates a differentiable representation of the manifold's local connectivity.

2.  **Structural Entropy Calculation ($H_1, H_2$)**:
    - **1D Entropy ($H_1$)**: Measures the baseline uncertainty of the graph's node distribution.
    - **2D Entropy ($H_2$)**: Measures the uncertainty of the graph partitioned into communities (e.g., "rooms" or "behavioral modes").
    - **Implementation**: The graph is partitioned at a "Temporal Bottleneck" (points of maximum latent variance $\|z_{t+1} - z_t\|$). 
    - **Loss**: The SIT loss minimizes $H_2$, forcing the encoder to produce latents that naturally cluster into distinct, reachable regions.

3.  **Integration with TD-MPC2**:
    - The SIT loss is added as an auxiliary objective to the TD-MPC2 world model (alongside consistency and reward prediction).
    - **Annealing Strategy**: $SIT_{coef}$ is decayed (e.g., $0.1 \to 0.01$) to allow early structural discovery without destabilizing late-stage fine-tuning.

### B. Controller (Stable-PPO vs. Jumpy-MPC)
- **Stable-PPO**: Dual-head Actor-Critic with decoupled `actor_logstd` (CleanRL style).
- **TD-MPC2 Configuration**:
    - **MPPI Planning**: High-frequency local control using the SIT-regularized latent model.
    - **Jumpy-MPC (Long Horizon)**: The Manager plans over the macro-states (SIT communities) discovered by the graph entropy analysis.
    - **SIT Potentials**: Intrinsic rewards are provided for "crossing boundaries" between discovered topological regions.

---

## 3. Pixel-Based Architecture (Future v3)
**Objective**: Scale to Atari/DMLab without an expensive RSSM.

### A. Spatial Feature Extraction
- **Input**: $84 \times 84$ Pixels.
- **Encoder**: Shallow CNN or Frozen Pre-trained ViT.
- **Discretization**: Vector Quantization (VQ-VAE style) to map pixels to a finite set of visual "tokens" ($V$).
- **SIT Graph**: Transitions between visual tokens form the adjacency matrix $A$ used for $H_2$ minimization.

### B. The Unified Director Implementation
The **Director** agent acts as the bridge between the high-level topological plan and low-level control.

1.  **Manager (High-Level)**:
    - Operates at a slow timescale (e.g., once every 8 steps).
    - Outputs a discrete **Skill Code** $w \in \{1 \dots N\}$.
    - Trained via PPO to maximize extrinsic reward $r_{ext}$ by sequencing skills.

2.  **Goal Autoencoder (VAE)**:
    - Maps the discrete Skill Code $w$ to a **Latent Goal Vector** $g$ in the world model's $z$-space.
    - Ensures that the Manager only proposes goals that the World Model considers "reachable" based on the training distribution.

3.  **Worker (Low-Level)**:
    - Receives $(s_t, g)$ at every environment step.
    - Trained to reach $g$ via an **Intrinsic Goal-Reaching Reward**:
      $$r_{int} = -\text{dist}(z_t, g)$$
    - Action Space: Environment-specific (Discrete or Continuous).

---

## 4. Analysis of Failure Modes

### Gap 1: Discretization Artifacts
- **Problem**: K-Means clusters are Voronoi cells. If an agent "vibrates" at the boundary of two clusters, it creates hundreds of "fake" transitions in the GraphTracker.
- **Analysis**: The graph becomes densely connected noise, leading the SIT algorithm to find no meaningful structure (or too many micro-clusters).

### Gap 2: Exploration Mismatch
- **Problem**: Continuous PPO using a Normal distribution is biased toward the center (0.0 torque). Environments like `Pendulum` require high-energy torque at the extremes to swing up.
- **Analysis**: Standard PPO enters a local minimum where it stays near vertical-down to minimize velocity penalties.

---

## 4. Integration Strategy: Modifying Director Source vs. Scratch

### Recommendation: Modular Integration
It is **not recommended** to develop directly inside the production `Director` source code if it was designed purely for discrete maps.

| Strategy | Pros | Cons |
| :--- | :--- | :--- |
| **Direct Modification** | Reuses existing tensorboard/logging. | High risk of breaking validated discrete logic. |
| **Modular Refactor (Recommended)** | Abstract the "State Encoder" and "Actor" classes. | Requires initial refactoring effort. |

### Proposed Modification Path:
1. **Abstract the Observer**: Create a `StateWrapper` that can either return a raw ID (Discrete) or a Cluster ID (Continuous).
2. **Policy Agnosticism**: Allow the Worker to swap between `DiscretePolicy` (Softmax) and `ContinuousPolicy` (Normal/Beta).
3. **Temporal Abstraction**: In continuous domains, the Director's goal must persist for several steps (e.g., 10-20 steps) to allow the Worker time to reach a different latent cluster.
