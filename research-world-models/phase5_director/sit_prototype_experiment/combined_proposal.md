# Combined Proposal: SIT-Director - A Unified Hierarchical World Model

## 1. Executive Summary: The Evolution from Director to SIT-Director
This proposal synthesizes the vision of a unified hierarchical reinforcement learning (HRL) architecture ([proposal.md](proposal.md)) with the rigorous mathematical grounding of **Structural Information Theory (SIT)** ([PROPOSAL.md](PROPOSAL.md) and [SIT_Director_Proposal.md](SIT_Director_Proposal.md)). We propose replacing the unstructured "Goal Autoencoder" of the original Director (Hafner et al., 2022) with a **SIT-optimized Encoding Tree**, creating a principled framework for discovering environment hierarchy across discrete, continuous, and pixel-based domains.

## 2. Similarities and Key Differences

| Feature | `proposal.md` (Latest) | `PROPOSAL.md` (Two-Level) | `SIT_Director_Proposal.md` (Full Tree) |
| :--- | :--- | :--- | :--- |
| **Core Goal** | Unified architecture (Grid $\to$ Pixels) | Replace Euclidean subgoals with modules | Adaptive abstraction & skill discovery |
| **Hierarchy** | 2-level (Manager/Worker) | 2-level (States $\to$ Modules) | Multi-level (Encoding Tree $T$) |
| **World Model** | SIT Graph as World Model | RSSM + 1D Structural Entropy | RSSM + $K$-D Encoding Tree |
| **Discretization**| KMeans / VQ-VAE / LSEnet | Latent Module Network $\mathcal{C}_\psi$ | Dynamic Tree $T$ (Merge/Split/Lift) |
| **Reward** | Intrinsic "In-Cluster" indicator | Boundary "Cut" crossing | Transition probability gradient |

---

## 3. Contribution vs. JMLR / Prior Art
Our work extends the foundations of Structural Information Theory (e.g., JMLR 2026) by addressing their primary limitation: **the requirement of a known, discrete MDP.**

### 3.1. Imagination-Based Abstraction
Unlike traditional SIT, which operates on static graph datasets, we introduce the **Structural World Model**.
- **Learning from Imagination**: We do not require a pre-calculated transition matrix. Instead, we use the differentiable imagination of a World Model (RSSM or TD-MPC2) to sample potential futures. SIT is then applied to the *imagined* transition graph to discover boundaries before the agent ever encounters them in reality.
- **Encoding Tree as World Model**: The Encoding Tree $T$ serves as more than just a cluster; it becomes an **Abstract World Model**. We learn high-level dynamics $P(\alpha_{t+1} | \alpha_t, w)$ where $\alpha$ are nodes in the tree, allowing for "jumpy" planning that bypasses micro-step physics.

---

## 4. Assumptions and Flexibility

### 4.1. Beyond Demonstrations (No SISL Needed)
While works like **SISL** require expert demonstration videos to segment tasks and learn hierarchies, SIT-Director is **fully unsupervised**. It discovers task segments by minimizing structural entropy in the self-supervised latent space. If a "doorway" exists in the latent dynamics, SIT will find it as a bottleneck, regardless of whether a human has shown how to use it.

### 4.2. Multi-Modal Interaction (Beyond SISA Pixels)
Previous architectures like **SISA** are often coupled to visual feature extraction. Our **Unified Discretizer** interface allows the same SIT-Director logic to handle:
- **Type A (Discrete)**: Raw state IDs.
- **Type B (Continuous)**: Latent vectors from MiniBatchKMeans or Hyperbolic Encoders.
- **Type C (Pixels)**: VQ-tokens from CNN encoders.
The topological analysis remains invariant to the input modality.

---

## 5. Research Goals (The ABCs of SIT-Director)

### A. Adaptive Temporal Abstraction
We solve the "Fixed $K$" problem (e.g., Manager acts every 8 steps) by using SIT boundaries. A Manager only issues a new goal when the latent state $z_t$ crosses a discovered community boundary (the "cut" $g_\alpha$), allowing skills to last as long as the environment's geometry dictates.

### B. Jumpy Abstract World Models
Once the Encoding Tree $T$ is formed, we plan directly at the macro-state level. This enables the Manager to sequence high-level transitions (e.g., "Enter Kitchen" $\to$ "Open Fridge") without simulating the individual joint movements required for each.

### C. Unsupervised Skill Discovery
A "skill" is defined mathematically as a policy that navigates to a specific module boundary in the tree $T$. By maximizing Decoding Information $D_T(G)$, we discover "Natural Joints" (bottlenecks) that serve as intrinsic targets for the Worker, effectively performing **Options Discovery in Latent Space**.

---

## 6. Implementation: The Unified Framework
We provide a unified training loop that integrates state abstraction and action hierarchy in a single differentiable pass:

1. **The SIT-Director Loop**:
   - **Rollout**: Collect transitions $(s_t, a_t, s_{t+1})$ into a buffer.
   - **Imagination**: The World Model generates a latent graph $G$ from the current policy's predicted transitions.
   - **Tree Optimization**: Update the Encoding Tree $T$ to minimize $H_T(G)$ using local operators (Merge, Split, Lift).
   - **Hierarchical Policy**:
     - **Manager** optimizes extrinsic reward by selecting nodes in $T$.
     - **Worker** optimizes intrinsic reward (boundary alignment) to reach the Manager's target.
   - **Unified Update**: All components (World Model, Tree, and Policies) are updated in a single training iteration, ensuring the hierarchy evolves as the model's understanding of physics improves.
