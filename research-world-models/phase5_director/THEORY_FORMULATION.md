# Theory Formulation: SIT-Director vs. Classical SIT
*Translating JMLR SIT into Latent World Models*

To build a strong, theoretically sound paper, we must clearly define classical Structural Information Theory (as detailed in the JMLR decision-making paper) and then propose our own formal mathematical extension that differentiates our work.

---

## Part 1: Explaining the JMLR SIT Theory (In Plain English)
The classical SIT theory for decision-making assumes the environment is a known or fully observable graph (MDP). Here is what their math actually means in plain words:

### 1. The Environment Graph $G = (V, E, W)$
*   **Math:** $V$ are states, $E$ are transitions, and $W_{ij}$ is the transition probability from state $i$ to $j$.
*   **Plain English:** Imagine the environment as a web of rooms (nodes). The thickness of the string connecting two rooms (weight) is how often the agent naturally wanders from one room to the other.

### 2. 1-Dimensional Structural Entropy ($H_1(G)$)
*   **Math:** $H_1(G) = - \sum_{i \in V} \frac{d_i}{2m} \log_2 \frac{d_i}{2m}$
    *(where $d_i$ is the volume/degree of node $i$, and $2m$ is the total volume).*
*   **Plain English:** If we drop a blindfolded agent into the environment and let it wander randomly indefinitely, how hard is it to guess which room it is currently in? This measures the *total baseline chaos* or *uncertainty* of the raw environment without any hierarchical grouping.

### 3. Tree Structural Entropy ($H_T(G)$)
*   **Math:** $H_T(G) = \sum_{\alpha \in T, \alpha \neq \lambda} - \frac{g_\alpha}{2m} \log_2 \frac{V_\alpha}{V_{\alpha^-}}$
    *(where $g_\alpha$ is the "cut", $V_\alpha$ is the module volume, and $V_{\alpha^-}$ is the parent module's volume).*
*   **Plain English:** Suppose we group the rooms into "Houses" (modules, $\alpha$). $V_\alpha$ is how much time the agent spends inside a specific House. The "cut" $g_\alpha$ is how often the agent steps *out* of the House's front doors. 
This formula asks: Once you know which House the agent is in, how much chaos/uncertainty is *still left* regarding exactly which specific room it is in? If we group the rooms smartly (so the agent rarely leaves the House), the remaining uncertainty ($H_T$) is very low.

### 4. Decoding Information ($D_T(G)$)
*   **Math:** $D_T(G) = H_1(G) - H_T(G)$
*   **Plain English:** **The "Aha!" Metric.** It is the difference between the total baseline chaos and the remaining chaos after grouping. It measures exactly how much "order" or "knowledge" our grouping strategy successfully mined out of the environment. Finding the best abstraction (the optimal tree) means maximizing this number.

---

## Part 2: Proposing Our New Theory (Imagination-Based SIT)
**The Core Difference:** The JMLR paper applies those formulas to the *ground-truth physical states* ($\mathcal{S}$). In contrast, our agent does not have access to a clean, finite state space. It only has pixels. Therefore, our theory must operate over the **Internal Belief Space** (latent representations) of a Neural Network. 

Here is how we formally propose our new theoretical framework for the method section:

### Definition 1: The Imagination Graph ($\mathcal{G}_\phi$)
Instead of measuring a static environment, we construct a graph over the World Model’s imagination.
Let $\mathcal{Z}$ be the latent state space of the RSSM (e.g., categorical vectors). We define the Latent Imagination Graph $\mathcal{G}_\phi = (\mathcal{Z}, \mathcal{P}_\phi)$ parameterized by the neural network weights $\phi$.
*   **The Nodes (Vertices):** Sampled latent belief states $z_t \sim q_\phi(z_t | x_{\le t}, a_{\le t})$.
*   **The Edges (Weights):** The prior transition dynamics modeled by the RSSM, marginalized over the worker policy $\pi_\theta$:
    $$W(z_i, z_j) = \mathbb{E}_{a \sim \pi_\theta(\cdot|z_i)} \left[ P_\phi(z_j | z_i, a) \right]$$
**Contribution:** We are the first to define structural entropy over a *learned, continuous/stochastic imagination space* rather than a physical MDP.

### Definition 2: The Latent Encoding Tree ($\mathcal{T}_\psi$)
We replace Director’s arbitrary "Goal Autoencoder" with a differentiable hierarchy. We formulate a neural tree-builder $\mathcal{T}_\psi(z)$ that maps any continuous/discrete latent state $z$ into a hierarchical module assignment $\alpha$. 
To optimize $\psi$, we formulate a differentiable loss function based on the JMLR decoding information:
$$ \mathcal{L}_{tree}(\psi) = - D_{\mathcal{T}_\psi}(\mathcal{G}_\phi) = H_{\mathcal{T}_\psi}(\mathcal{G}_\phi) - H_1(\mathcal{G}_\phi) $$
By minimizing this loss, our neural network natively discovers the natural bottlenecks (doors, subgoals) inside the world model's imagination.

### Definition 3: Structural Hierarchical Planning (SIT-Actor-Critic)
In Hafner's Director, the Manager sets an ungrounded feature vector as a goal, and the Worker maximizes cosine similarity to it. We redefine this mathematically using SIT boundaries.

**Manager Policy ($\pi_{mgr}$):**
Instead of predicting an arbitrary vector $g$, the Manager selects a *Target Module* $\alpha \in \mathcal{T}_\psi$. 
Its objective is to maximize extrinsic reward, plus an intrinsic bonus for discovering new decoding information (structural curiosity):
$$ J_{mgr} = \mathbb{E} \left[ \sum \gamma^t \left( r_t^{ext} + \eta \Delta D_{\mathcal{T}_\psi}(\mathcal{G}_\phi) \right) \right] $$

**Worker Policy ($\pi_{wkr}$):**
The Worker receives the target module $\alpha$ from the Manager. Its intrinsic reward is formulated as the probability flux of crossing the structural boundary cut $g_\alpha$ into the target module:
$$r_t^{int} = \mathbb{P}_\phi(z_{t+1} \in \alpha \mid z_t \notin \alpha, a_t)$$
**Contribution:** The worker is no longer "chasing a vector." It is optimizing the transition probability of crossing an information-theoretic boundary in the latent space.

### Definition 4: Jumpy Latent Dynamics
Because SIT legally bounds the internal entropy of a module, we can define a multi-scale transition model. Instead of the RSSM rolling out step-by-step $P(z_{t+1}|z_t, a)$, we define the Abstract Transition Model:
$$ P(\alpha_{next} | \alpha_{current}, \pi_{mgr}) $$
**Contribution:** This mathematically proves we can skip time-steps. The Manager plans over the macro-graph of modules $\alpha$, yielding orders of magnitude faster planning than Danijar's Dreamer/Director which unrolls every micro-step.

---

### Summary for the Paper
By formulating these equations, your paper will state:
*"While classical SIT analyzes concrete Markov Decision Processes (JMLR 2026), SIT-Director is the first framework to embed structural entropy minimization natively into the latent dynamics of an end-to-end differentiable World Model. By replacing the generic Goal Autoencoder with the Latent Encoding Tree $\mathcal{T}_\psi$, our Manager and Worker mathematically optimize boundary crossings rather than arbitrary feature distances, achieving true unsupervised skill discovery from pixels."*