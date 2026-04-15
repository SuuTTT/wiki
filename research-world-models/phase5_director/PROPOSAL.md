# Proposal: Structural Information Theory for Hierarchical Latent World Models (SIT-Director)

## 1. Abstract
Recent advances in model-based reinforcement learning, such as the Director architecture, achieve hierarchical planning by jointly training a Manager policy and a Worker policy within a learned latent world model. However, Director relies on a "Goal Autoencoder" that defines subgoals as continuous vectors, using arbitrary Euclidean or cosine distance metrics. This approach lacks topological grounding and struggles to discover natural bottlenecks in the environment. 

We propose **SIT-Director**, a novel framework that integrates Structural Information Theory (SIT) natively into the differentiable imagination space of a World Model. Instead of a continuous goal vector, we formulate goal-reaching as transitioning across topological boundaries (modules) in the latent transition graph. To constrain the scope of this initial work, we restrict our abstraction to a **two-level hierarchy** (base latent states and macro-modules), reserving multi-level encoding trees for future work. By minimizing 1-dimensional structural entropy over the imagined transition graph, we achieve mathematically rigorous, unsupervised subgoal discovery purely from pixels.

---

## 2. Motivation
In Director (Hafner et al., 2022), the Manager selects a feature vector $g$ in the latent space, and the Worker maximizes its cosine similarity to $g$. This form of goal-reaching assumes the latent space is a flat, homogeneous manifold. In environments with hard boundaries (e.g., navigating from room A to room B via a narrow door), Euclidean distance in latent space does not reflect the true dynamic topological distance.

Structural Information Theory (JMLR, 2026) provides a mathematical framework for finding the natural "cuts" or boundaries in an environment, maximizing the *Decoding Information*. However, classical SIT operates on a known, discrete Markov Decision Process (MDP). Our core theoretical contribution is mapping SIT principles into the continuous, learned imagination of a Neural World Model.

---

## 3. Theoretical Framework (Two-Level Hierarchy)

We adapt SIT to define a single layer of abstraction over the latent space.

### 3.1. The Latent Imagination Graph ($\mathcal{G}_\phi$)
Let $\mathcal{Z}$ be the continuous/categorical latent state space modeled by a Recurrent State Space Model (RSSM) with parameters $\phi$. We define the Latent Imagination Graph $\mathcal{G}_\phi = (\mathcal{Z}, \mathcal{P}_\phi)$.
*   **Nodes (Vertices):** Latent belief states $z_t \sim q_\phi(z_t | x_{\le t}, a_{\le t})$.
*   **Edges (Weights):** The temporal transition dynamics prior, aggregated over the current worker policy $\pi_\theta$:
    $$W(z_i, z_j) = \mathbb{E}_{a \sim \pi_\theta(\cdot|z_i)} \left[ P_\phi(z_j | z_i, a) \right]$$

### 3.2. Two-Level Structural Entropy
Instead of a full multi-level encoding tree, we partition the latent space $\mathcal{Z}$ into a discrete set of modules (communities) $\mathcal{M} = \{\alpha_1, \alpha_2, \dots, \alpha_K\}$.
*   **1-D Entropy ($H_1$):** The baseline uncertainty of a random walk in the latent imagination.
    $$H_1(\mathcal{G}_\phi) = - \sum_{z \in \mathcal{Z}} \frac{d_z}{2m} \log_2 \frac{d_z}{2m}$$
*   **Partition Entropy ($H_{\mathcal{M}}$):** The remaining uncertainty after state space partitioning. It is scaled by the boundary "cuts" $g_\alpha$ (the probability mass of leaving module $\alpha$) and the internal volume $V_\alpha$ (the probability mass of staying in $\alpha$).
    $$H_{\mathcal{M}}(\mathcal{G}_\phi) = \sum_{\alpha \in \mathcal{M}} - \frac{g_\alpha}{2m} \log_2 \frac{V_\alpha}{2m}$$
*   **Decoding Information ($D_{\mathcal{M}}$):** The objective is to find a neural partition function $\mathcal{C}_\psi(z) \to \alpha$ that maximizes the extracted structure:
    $$D_{\mathcal{M}}(\mathcal{G}_\phi) = H_1(\mathcal{G}_\phi) - H_{\mathcal{M}}(\mathcal{G}_\phi)$$

---

## 4. Proposed Method: SIT-Director

We replace Director's Goal Autoencoder with the **Latent Module Network** $\mathcal{C}_\psi$.

### 4.1. The Latent Module Network
The network $\mathcal{C}_\psi(z)$ outputs a categorical distribution over $K$ available modules. It is trained to minimize the structural entropy of the imagination graph:
$$ \mathcal{L}_{module}(\psi) = H_{\mathcal{M}_\psi}(\mathcal{G}_\phi) $$
This naturally forces $\mathcal{C}_\psi$ to draw boundaries at the dynamical bottlenecks of the latent space, defining natural subgoals.

### 4.2. Structural Hierarchical Planning
With the latent space cleanly partitioned into topological modules, we redefine the Manager and Worker reinforcement learning objectives.

**The Manager Policy ($\pi_{mgr}$):**
Operating at a slow time-scale, the Manager observes the current module $\alpha_{t} = \mathcal{C}_\psi(z_t)$. Instead of predicting a continuous vector, the Manager predicts a discrete *Target Module* $\alpha_{target}$.
$$ \pi_{mgr}(\alpha_{target} \mid z_t) $$
The Manager is trained to maximize extrinsic environmental rewards. 

**The Worker Policy ($\pi_{wkr}$):**
Operating at a fast time-scale, the Worker receives $z_t$ and the Manager's $\alpha_{target}$. The Worker's intrinsic reward is no longer a cosine similarity to a vector. Instead, it is rewarded for crossing the structural boundary cut $g_{\alpha}$ into the target module:
$$ r_t^{int} = \begin{cases} 
1 & \text{if } \mathcal{C}_\psi(z_{t}) \neq \alpha_{target} \text{ and } \mathcal{C}_\psi(z_{t+1}) = \alpha_{target} \\
0 & \text{otherwise}
\end{cases} $$
*(Alternatively, a dense reward based on the transition probability gradient toward $\alpha_{target}$ can be used).*

---

## 5. Scope and Future Work
By restricting the SIT integration to a two-level hierarchy (states $\to$ modules), we bypass the combinatorial complexity of dynamic tree-building while still completely replacing Director's ungrounded Euclidean subgoals. 

**Future Work:** 
Once the two-level abstraction proves that structurally grounded bottlenecks dramatically improve sample efficiency and exploration compared to Director, future work will naturally extend $\mathcal{C}_\psi$ into a fully differentiable *Latent Encoding Tree*. This will allow for multi-level hierarchical planning, enabling $n$-step temporal jumps and infinite-horizon goal formulation.