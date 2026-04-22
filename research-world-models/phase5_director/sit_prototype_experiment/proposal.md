# Proposal: SIT-Director as a Unified Hierarchical Framework

## 1. Vision
To create a single, unified hierarchical reinforcement learning (HRL) architecture that scales from discrete gridworlds to continuous control and raw pixels. The core innovation is replacing the resource-heavy, parametric world model (RSSM) with a **Structural Information-Theoretic (SIT) Graph**, which acts as a non-parametric abstraction of the environment's topology.

## 2. The Unified Theory: SIT as a World Model
In traditional Model-Based RL (e.g., Dreamer), a world model predicts $s_{t+1}$ given $s_t, a_t$. 
In the **SIT-Director** framework, we define a "World Model" not as a predictive transition function, but as a **Topological Manifold**:
1.  **Nodes ($V$):** Represent distinct "states" or "regions" (Discrete IDs, K-Means Clusters, or VAE Latents).
2.  **Edges ($E$):** Represent the *reachability* between these regions.
3.  **Communities ($C$):** Represent "Macro-States" or "Rooms" discovered via Structural Information Theory.

By maximizing the structure of this graph, the agent discovers the "true" hierarchy of the environment without needing to learn complex dynamics.

## 3. Input-Agnostic Abstraction Layers
The framework handles different input types by swapping only the **Spatial Discretizer**:

### Type A: Discrete & Small State (e.g., FrozenLake)
- **Discretizer**: Identity Mapper (State ID $\rightarrow$ Node ID).
- **Graph**: Perfect adjacency matrix.
- **Goal**: Direct state reaching.

### Type B: Continuous State Space (e.g., Pendulum, LunarLander, MuJoCo)
- **Discretizer**: Online `MiniBatchKMeans`, **LSEnet-Style Hyperbolic Encoder**, or **TD-MPC2 Latent Model**.
- **Graph**: Manifold approximated by cluster-to-cluster transitions or **latent-transition graph** in TD-MPC2.
- **Goal**: Latent cluster reaching or **Topological Anchor Point Navigation** (Jumpy-MPC).

### Type C: Pixels (e.g., Atari, DeepMind Lab)
- **Discretizer**: Pre-trained or concurrently learned VAE / CNN-Encoder + SIT (**Differentiable Structural Information**).
- **Graph**: Abstract feature-space topology.
- **Goal**: Visual landmark reaching.

## 4. Hierarchical Logic (The Unified Director)
Regardless of the input type, the high-level logic remains identical:
1.  **Manager**: Observes the current "Node" and selects a target "Community" (SIT Cluster) as a long-term goal. 
    - *Advanced*: In continuous manifolds, the target is a branch in an **LSEnet Partitioning Tree**.
2.  **Worker**: Receives the target Community and earns an intrinsic reward for moving the system into any "Node" belonging to that Community.
3.  **Optimization**: PPO-based actor-critic for both levels, with the "Frozen SIT" trick for stabilization.

## 5. References
- **Director Algorithm**: Hafner et al., "Deep Hierarchical Planning from Pixels" (2022). [Link](https://ar5iv.labs.arxiv.org/html/2206.04114)
- **LSEnet (DSI)**: Sun et al., "LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering" (2024). [Link](https://ar5iv.labs.arxiv.org/html/2405.11801)
- **TD-MPC2**: Hansen et al., "TD-MPC2: Scalable Model-Based Reinforcement Learning" (2023). [Link](https://arxiv.org/abs/2310.16828)
- **SIT Theory**: Li & Pan, "Structural Information and Dynamical Complexity of Networks" (2016). [Link](https://nature.com/articles/srep23214)
- **Structural Entropy**: An et al., "The Differentiable Structural Information for Complex Networks" (2023).
- [ ] **Phase 1**: Refactor `Director` source to support a `StateAbstraction` interface.
- [ ] **Phase 2**: Implement `KMeansDiscretizer` for Continuous Control. Add `actor_logstd` decoupling.
- [ ] **Phase 3**: Validate on Pendulum-v1 (Continuous) and 8x8 FrozenLake (Discrete) within the same script structure.
- [ ] **Phase 4**: (Optional/Future) Add `CNNEncoder` for pixel-based benchmarks.
