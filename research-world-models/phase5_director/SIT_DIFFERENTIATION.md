# SIT-Director Novelty and Method Differentiation
*Addressing Structural Information Principles for Decision Making (JMLR 2026)*

## 1. Using the JMLR Seminal Paper Theory in Our Method Section
Yes, we **can and must** use the mathematical theory established in the JMLR paper (e.g., Encoding Trees, 1D/T-Dimensional Structural Entropy, $H_T(G)$, and Decoding Information). However, they belong in a **Preliminaries / Background** section, rather than our core contribution.

**How to structure it:**
*   **Section 3: Preliminaries:** Introduce *Structural Information Principles for Decision Making* (cite JMLR). Define the environment graph $G$, random walk distribution, and equation for computing $H_T(G)$ over an MDP. Also, introduce Dreamer/Director's latent imagination objective.
*   **Section 4: Our Method (SIT-WM / SIT-Director):** Introduce *how we apply* these theoretical principles specifically to end-to-end, high-dimensional Continuous/Discrete Latent World Models (RSSM). The JMLR equations serve as our **Optimization Objective** rather than our contribution.

## 2. How to Avoid the Same Contribution
If the JMLR paper defines "SIT for Decision Making," it likely approaches RL by constructing graphs over the true state space $S$ (tabular or straightforward continuous states) or assumes an explicit model of transition probabilities. 

To ensure our research is highly distinct and novel, we must carve out our niche at the intersection of **Latent Abstract World Models** and **Pixel-based Hierarchical Planning**.

### Novelty Angle 1: From Real Trajectories to Latent Imagination
*   **Their Approach:** Likely constructs a graph $G$ from the raw state-transition tuples $(s_t, s_{t+1})$ drawn from an explicit MDP or replay buffer.
*   **Our Novelty:** We operate purely on the **imagined latent space ($z_t$)** of an RSSM. We define the vertices of $G$ as recurrent latent states, and the directed edges as the predicted transition probabilities $P(z_{t+1}|z_t)$ computed by the World Model's dynamics predictor. This makes our clustering differentiable and enables Structural Entropy partitioning over continuous *visual representations* (pixels).

### Novelty Angle 2: Replacing the "Goal Autoencoder"
*   **Their Approach:** Discovers macro-states ($\alpha$) independently to solve long-horizon exploration.
*   **Our Novelty:** We specifically solve the structural flaw in Danijar's **Director** architecture. Instead of Director's arbitrary "Goal Autoencoder" (which merely forces states into a categorical bottleneck with no guarantees about causality or environment geometry), we use the Encoding Tree from SIT. *Our Encoding Tree is our Goal Autoencoder.*

### Novelty Angle 3: Variable-Horizon Jumpy Planning
*   **Their Approach:** Discovers options/skills but keeps the base execution tied to standard MDP time-stepping.
*   **Our Novelty:** Director forces a Manager update exactly every $K=8$ steps. By identifying the boundary cut $g_\alpha$ using SIT, our Manager's control interval becomes completely adaptive. We update the Goal *only* when the Worker successfully transitions across a SIT module boundary. Furthermore, we can train a **Jumpy Dynamics Model** to predict $P(\alpha_{t+K} | \alpha_t)$, completely skipping pixel-level RSSM rollouts during abstract planning.

## 3. Summary of Our Unique Contributions for the Paper
If you are writing the Introduction, frame our contributions exactly like this:
1.  We propose **SIT-WM (Structural Information-Theoretic World Models)**, an extension of the RSSM that continuously maps visual observations into a latent space and simultaneously builds an optimal Encoding Tree to minimize structural entropy.
2.  We introduce **SIT-Director**, a robust hierarchical planning framework that replaces the heuristic Goal Autoencoder from Hafner et al. with an SIT Module. It allows goal generation derived mathematically from the environment's intrinsic geometry rather than arbitrary bottlenecks.
3.  We exhibit the first application of SIT Principles to end-to-end pixel-based Reinforcement Learning via parallel Manager/Worker actor-critic loops trained completely inside latent imagination.